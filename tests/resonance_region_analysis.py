from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter
from typing import Iterable, Literal, Sequence

import numpy as np

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "auto_preconditioner_mpl"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from benchmark_real_data import (  # noqa: E402
    CompressedDataset,
    DEFAULT_DATA_DIR,
    TargetKind,
    build_impedance_matrix,
    build_rhs,
    build_ridge_preconditioner_factory,
    default_preconditioners,
    default_solvers,
    load_compressed_dataset,
    resolve_data_dir,
)


ResponseMetric = Literal["relative_gain", "response_norm", "max_current_abs"]


@dataclass(frozen=True)
class ResponseRecord:
    omega_rad_s: float
    response_norm: float
    rhs_norm: float
    relative_gain: float
    max_current_abs: float
    direct_residual: float
    solve_method: str
    diag_impedance_abs: float
    diag_reactance: float


@dataclass(frozen=True)
class ExtremumRecord:
    kind: str
    rank: int
    index: int
    omega_rad_s: float
    response_metric: str
    response_value: float
    response_norm: float
    relative_gain: float
    max_current_abs: float
    direct_residual: float
    neighborhood_start_omega_rad_s: float
    neighborhood_end_omega_rad_s: float
    neighborhood_size: int


@dataclass(frozen=True)
class FrequencySolveRecord:
    omega_rad_s: float
    preconditioner: str
    solver: str
    info: int
    converged: bool
    n_iter: int
    residual: float
    preconditioner_time_s: float
    solve_time_s: float
    total_time_s: float
    response_norm: float
    relative_gain: float
    max_current_abs: float


@dataclass(frozen=True)
class NeighborhoodSummaryRecord:
    point_kind: str
    point_rank: int
    omega_rad_s: float
    preconditioner: str
    solver: str
    center_iterations: int
    center_residual: float
    neighborhood_size: int
    neighborhood_iterations_mean: float
    neighborhood_iterations_std: float
    neighborhood_iterations_min: int
    neighborhood_iterations_max: int
    neighborhood_residual_mean: float
    neighborhood_residual_std: float
    neighborhood_residual_min: float
    neighborhood_residual_max: float
    center_minus_neighborhood_iterations: float
    center_to_neighborhood_residual_ratio: float


@dataclass(frozen=True)
class ResonanceAnalysisResult:
    dataset: CompressedDataset
    omegas: tuple[float, ...]
    response: tuple[ResponseRecord, ...]
    extrema: tuple[ExtremumRecord, ...]
    solve_records: tuple[FrequencySolveRecord, ...]
    neighborhood_summaries: tuple[NeighborhoodSummaryRecord, ...]
    skipped: tuple[str, ...]
    output_dir: Path


def theoretical_resonance(dataset: CompressedDataset) -> tuple[float, float]:
    omega0 = 1.0 / np.sqrt(dataset.L * dataset.C)
    return float(omega0), float(omega0 / (2.0 * np.pi))


def select_analysis_omegas(
    dataset: CompressedDataset,
    *,
    res_min: float,
    res_max: float,
    limit: int | None = None,
) -> np.ndarray:
    omegas = dataset.selected_freqs(res_min=res_min, res_max=res_max, limit=None)
    if limit is None or limit >= len(omegas):
        return omegas
    indices = np.linspace(0, len(omegas) - 1, num=limit, dtype=int)
    return omegas[np.unique(indices)]


def _relative_residual(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(b)), 1e-30)
    return float(np.linalg.norm(b - A @ x) / denom)


def solve_exact_response(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float, str]:
    try:
        x = np.linalg.solve(A, b)
        method = "solve"
    except np.linalg.LinAlgError:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        method = "lstsq"
    return x, _relative_residual(A, x, b), method


def scan_frequency_response(
    dataset: CompressedDataset,
    omegas: Sequence[float],
) -> list[ResponseRecord]:
    records: list[ResponseRecord] = []
    for omega in omegas:
        Z = build_impedance_matrix(dataset, float(omega))
        rhs = build_rhs(dataset, float(omega))
        x, residual, solve_method = solve_exact_response(Z, rhs)
        response_norm = float(np.linalg.norm(x))
        rhs_norm = float(np.linalg.norm(rhs))
        diag = dataset.R - 1j * omega * dataset.L - 1.0 / (1j * omega * dataset.C)
        records.append(
            ResponseRecord(
                omega_rad_s=float(omega),
                response_norm=response_norm,
                rhs_norm=rhs_norm,
                relative_gain=response_norm / max(rhs_norm, 1e-30),
                max_current_abs=float(np.max(np.abs(x))),
                direct_residual=residual,
                solve_method=solve_method,
                diag_impedance_abs=float(np.abs(diag)),
                diag_reactance=float(np.imag(diag)),
            )
        )
    return records


def _response_value(record: ResponseRecord, metric: ResponseMetric) -> float:
    if metric == "relative_gain":
        return record.relative_gain
    if metric == "response_norm":
        return record.response_norm
    return record.max_current_abs


def _local_extrema_candidates(values: np.ndarray, *, kind: str) -> list[int]:
    n = int(values.size)
    if n == 0:
        return []
    if n == 1:
        return [0]

    candidates: list[int] = []
    for idx in range(1, n - 1):
        left = values[idx - 1]
        center = values[idx]
        right = values[idx + 1]
        if kind == "peak":
            if center >= left and center > right:
                candidates.append(idx)
        else:
            if center <= left and center < right:
                candidates.append(idx)
    return candidates


def _select_separated_indices(
    candidates: Sequence[int],
    scores: np.ndarray,
    *,
    count: int,
    min_separation: int,
) -> list[int]:
    chosen: list[int] = []
    ordered = sorted(candidates, key=lambda idx: scores[idx], reverse=True)
    for idx in ordered:
        if all(abs(idx - prev) >= min_separation for prev in chosen):
            chosen.append(idx)
        if len(chosen) == count:
            break
    return chosen


def _fill_missing_extrema(
    selected: list[int],
    *,
    values: np.ndarray,
    count: int,
    min_separation: int,
    prefer_largest: bool,
) -> list[int]:
    if len(selected) >= count:
        return selected[:count]

    ordered = np.argsort(values)
    if prefer_largest:
        ordered = ordered[::-1]
    for idx in ordered.tolist():
        if idx in selected:
            continue
        if all(abs(idx - prev) >= min_separation for prev in selected):
            selected.append(int(idx))
        if len(selected) == count:
            break
    return selected[:count]


def find_response_extrema(
    response: Sequence[ResponseRecord],
    *,
    metric: ResponseMetric = "relative_gain",
    count_per_kind: int = 3,
    min_separation: int = 8,
    neighborhood_radius: int = 8,
) -> list[ExtremumRecord]:
    if count_per_kind <= 0:
        raise ValueError("count_per_kind must be positive")
    if neighborhood_radius < 0:
        raise ValueError("neighborhood_radius must be non-negative")

    values = np.asarray([_response_value(rec, metric) for rec in response], dtype=np.float64)
    effective_separation = max(int(min_separation), neighborhood_radius + 1, 1)

    peak_candidates = _local_extrema_candidates(values, kind="peak")
    trough_candidates = _local_extrema_candidates(values, kind="trough")

    peak_indices = _select_separated_indices(
        peak_candidates,
        values,
        count=count_per_kind,
        min_separation=effective_separation,
    )
    trough_indices = _select_separated_indices(
        trough_candidates,
        -values,
        count=count_per_kind,
        min_separation=effective_separation,
    )

    peak_indices = _fill_missing_extrema(
        peak_indices,
        values=values,
        count=count_per_kind,
        min_separation=effective_separation,
        prefer_largest=True,
    )
    trough_indices = _fill_missing_extrema(
        trough_indices,
        values=values,
        count=count_per_kind,
        min_separation=effective_separation,
        prefer_largest=False,
    )

    extrema: list[ExtremumRecord] = []
    for kind, indices in (("peak", peak_indices), ("trough", trough_indices)):
        sorted_indices = sorted(
            indices,
            key=lambda idx: values[idx],
            reverse=(kind == "peak"),
        )
        for rank, idx in enumerate(sorted_indices, start=1):
            left = max(0, idx - neighborhood_radius)
            right = min(len(response) - 1, idx + neighborhood_radius)
            rec = response[idx]
            extrema.append(
                ExtremumRecord(
                    kind=kind,
                    rank=rank,
                    index=idx,
                    omega_rad_s=rec.omega_rad_s,
                    response_metric=metric,
                    response_value=float(values[idx]),
                    response_norm=rec.response_norm,
                    relative_gain=rec.relative_gain,
                    max_current_abs=rec.max_current_abs,
                    direct_residual=rec.direct_residual,
                    neighborhood_start_omega_rad_s=response[left].omega_rad_s,
                    neighborhood_end_omega_rad_s=response[right].omega_rad_s,
                    neighborhood_size=right - left + 1,
                )
            )
    return sorted(extrema, key=lambda item: (item.kind, item.rank))


def _resolve_preconditioner_factories(
    dataset: CompressedDataset,
    omegas: np.ndarray,
    *,
    ignore_size_limits: bool,
    include_ridge: bool,
    ridge_reg: float,
    ridge_target: TargetKind,
    ridge_train_limit: int | None,
    ridge_max_n: int,
    preconditioner_names: Sequence[str] | None,
    verbose: bool,
) -> tuple[list, list[str]]:
    factories = list(default_preconditioners())
    skipped: list[str] = []

    if include_ridge:
        ridge_factory, ridge_message = build_ridge_preconditioner_factory(
            dataset,
            omegas,
            ridge_reg=ridge_reg,
            ridge_target=ridge_target,
            ridge_train_limit=ridge_train_limit,
            ridge_max_n=ridge_max_n,
            verbose=verbose,
        )
        if ridge_factory is None:
            skipped.append(ridge_message)
        else:
            factories.append(ridge_factory)
            if verbose:
                print(ridge_message)

    if preconditioner_names is not None:
        by_name = {factory.name: factory for factory in factories}
        selected: list = []
        for name in preconditioner_names:
            if name not in by_name:
                raise ValueError(f"Unknown preconditioner requested: {name!r}")
            selected.append(by_name[name])
        factories = selected

    supported: list = []
    for factory in factories:
        if ignore_size_limits or factory.supports(dataset.n):
            supported.append(factory)
        else:
            skipped.append(
                f"{factory.name}: skipped for N={dataset.n} (limit={factory.max_n})"
            )
    return supported, skipped


def _resolve_solvers(restart: int, solver_names: Sequence[str] | None) -> tuple:
    solvers = tuple(default_solvers(restart=restart))
    if solver_names is None:
        return solvers
    by_name = {solver.name: solver for solver in solvers}
    selected = []
    for name in solver_names:
        if name not in by_name:
            raise ValueError(f"Unknown solver requested: {name!r}")
        selected.append(by_name[name])
    return tuple(selected)


def run_frequency_solver_benchmark(
    dataset: CompressedDataset,
    omegas: Sequence[float],
    response: Sequence[ResponseRecord],
    *,
    tol: float = 1e-6,
    maxiter: int = 200,
    restart: int = 40,
    ignore_size_limits: bool = False,
    include_ridge: bool = True,
    ridge_reg: float = 1e-3,
    ridge_target: TargetKind = "pinv",
    ridge_train_limit: int | None = None,
    ridge_max_n: int = 64,
    preconditioner_names: Sequence[str] | None = None,
    solver_names: Sequence[str] | None = None,
    verbose: bool = True,
) -> tuple[list[FrequencySolveRecord], list[str]]:
    factories, skipped = _resolve_preconditioner_factories(
        dataset,
        np.asarray(omegas, dtype=np.float64),
        ignore_size_limits=ignore_size_limits,
        include_ridge=include_ridge,
        ridge_reg=ridge_reg,
        ridge_target=ridge_target,
        ridge_train_limit=ridge_train_limit,
        ridge_max_n=ridge_max_n,
        preconditioner_names=preconditioner_names,
        verbose=verbose,
    )
    solvers = _resolve_solvers(restart, solver_names)
    response_lookup = {float(item.omega_rad_s): item for item in response}

    records: list[FrequencySolveRecord] = []
    for factory in factories:
        if verbose:
            print(f"[{factory.name}] scanning {len(omegas)} angular frequencies...")
        for omega in omegas:
            Z = build_impedance_matrix(dataset, float(omega))
            rhs = build_rhs(dataset, float(omega))
            response_rec = response_lookup[float(omega)]

            t_pre0 = perf_counter()
            try:
                built = factory.build(Z)
            except Exception as exc:
                pre_time = perf_counter() - t_pre0
                skipped.append(
                    f"{factory.name}: build failed at omega={omega:.6e} with "
                    f"{type(exc).__name__}: {exc}"
                )
                for solver in solvers:
                    records.append(
                        FrequencySolveRecord(
                            omega_rad_s=float(omega),
                            preconditioner=factory.name,
                            solver=solver.name,
                            info=-1,
                            converged=False,
                            n_iter=0,
                            residual=float("inf"),
                            preconditioner_time_s=pre_time,
                            solve_time_s=0.0,
                            total_time_s=pre_time,
                            response_norm=response_rec.response_norm,
                            relative_gain=response_rec.relative_gain,
                            max_current_abs=response_rec.max_current_abs,
                        )
                    )
                continue

            pre_time = perf_counter() - t_pre0
            linear_operator = built.as_linear_operator()
            for solver in solvers:
                t_solve0 = perf_counter()
                try:
                    result = solver.solve(Z, rhs, linear_operator, tol, maxiter)
                    info = int(result.info)
                    converged = bool(result.converged)
                    n_iter = int(result.n_iter)
                    residual = _relative_residual(Z, result.x, rhs)
                except Exception:
                    info = maxiter
                    converged = False
                    n_iter = maxiter
                    residual = float("inf")
                solve_time = perf_counter() - t_solve0
                records.append(
                    FrequencySolveRecord(
                        omega_rad_s=float(omega),
                        preconditioner=factory.name,
                        solver=solver.name,
                        info=info,
                        converged=converged,
                        n_iter=n_iter,
                        residual=residual,
                        preconditioner_time_s=pre_time,
                        solve_time_s=solve_time,
                        total_time_s=pre_time + solve_time,
                        response_norm=response_rec.response_norm,
                        relative_gain=response_rec.relative_gain,
                        max_current_abs=response_rec.max_current_abs,
                    )
                )
    return records, skipped


def _neighbor_indices(center: int, radius: int, size: int) -> list[int]:
    left = max(0, center - radius)
    right = min(size - 1, center + radius)
    return list(range(left, right + 1))


def build_neighborhood_summaries(
    records: Sequence[FrequencySolveRecord],
    extrema: Sequence[ExtremumRecord],
    *,
    omegas: Sequence[float],
    neighborhood_radius: int,
) -> list[NeighborhoodSummaryRecord]:
    grouped: dict[tuple[str, str], dict[float, FrequencySolveRecord]] = {}
    for record in records:
        key = (record.preconditioner, record.solver)
        grouped.setdefault(key, {})[record.omega_rad_s] = record

    summaries: list[NeighborhoodSummaryRecord] = []
    omega_array = np.asarray(omegas, dtype=np.float64)
    for point in extrema:
        point_omega = float(point.omega_rad_s)
        indices = _neighbor_indices(point.index, neighborhood_radius, len(omega_array))
        neighbor_omegas = [float(omega_array[idx]) for idx in indices if idx != point.index]

        for (preconditioner, solver), mapping in sorted(grouped.items()):
            center = mapping.get(point_omega)
            if center is None:
                continue
            neighbors = [mapping[omega] for omega in neighbor_omegas if omega in mapping]
            if neighbors:
                neighbor_iterations = np.asarray([item.n_iter for item in neighbors], dtype=np.float64)
                neighbor_residuals = np.asarray([item.residual for item in neighbors], dtype=np.float64)
                iter_mean = float(np.mean(neighbor_iterations))
                iter_std = float(np.std(neighbor_iterations))
                iter_min = int(np.min(neighbor_iterations))
                iter_max = int(np.max(neighbor_iterations))
                res_mean = float(np.mean(neighbor_residuals))
                res_std = float(np.std(neighbor_residuals))
                res_min = float(np.min(neighbor_residuals))
                res_max = float(np.max(neighbor_residuals))
                res_ratio = (
                    float(center.residual / res_mean)
                    if np.isfinite(res_mean) and res_mean > 0.0
                    else float("inf")
                )
            else:
                iter_mean = float("nan")
                iter_std = float("nan")
                iter_min = center.n_iter
                iter_max = center.n_iter
                res_mean = float("nan")
                res_std = float("nan")
                res_min = center.residual
                res_max = center.residual
                res_ratio = float("nan")

            summaries.append(
                NeighborhoodSummaryRecord(
                    point_kind=point.kind,
                    point_rank=point.rank,
                    omega_rad_s=point_omega,
                    preconditioner=preconditioner,
                    solver=solver,
                    center_iterations=center.n_iter,
                    center_residual=center.residual,
                    neighborhood_size=len(neighbors),
                    neighborhood_iterations_mean=iter_mean,
                    neighborhood_iterations_std=iter_std,
                    neighborhood_iterations_min=iter_min,
                    neighborhood_iterations_max=iter_max,
                    neighborhood_residual_mean=res_mean,
                    neighborhood_residual_std=res_std,
                    neighborhood_residual_min=res_min,
                    neighborhood_residual_max=res_max,
                    center_minus_neighborhood_iterations=(
                        float(center.n_iter) - iter_mean if np.isfinite(iter_mean) else float("nan")
                    ),
                    center_to_neighborhood_residual_ratio=res_ratio,
                )
            )
    return summaries


def write_csv(path: Path, rows: Iterable[object]) -> None:
    rows_list = list(rows)
    if not rows_list:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows_list[0]).keys()))
        writer.writeheader()
        for row in rows_list:
            writer.writerow(asdict(row))


def _omega_to_mrad(omega: Sequence[float]) -> np.ndarray:
    return np.asarray(omega, dtype=np.float64) / 1e6


def _metric_label(metric: str) -> str:
    if metric == "relative_gain":
        return "Relative gain ||I|| / ||rhs||"
    if metric == "response_norm":
        return "Response norm ||I||"
    if metric == "max_current_abs":
        return "Max |I_i|"
    if metric == "iterations":
        return "Iterations"
    if metric == "residual":
        return "Relative residual"
    raise ValueError(f"Unknown metric: {metric!r}")


def _color_cycle() -> list[str]:
    return plt.rcParams["axes.prop_cycle"].by_key().get(
        "color",
        ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"],
    )


def plot_response_vs_omega(
    response: Sequence[ResponseRecord],
    extrema: Sequence[ExtremumRecord],
    *,
    metric: ResponseMetric,
    theoretical_omega: float,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    omegas = np.asarray([rec.omega_rad_s for rec in response], dtype=np.float64)
    values = np.asarray([_response_value(rec, metric) for rec in response], dtype=np.float64)
    ax.plot(_omega_to_mrad(omegas), values, marker="o", markersize=2.5, linewidth=1.0)

    peak_points = [point for point in extrema if point.kind == "peak"]
    trough_points = [point for point in extrema if point.kind == "trough"]
    if peak_points:
        ax.scatter(
            _omega_to_mrad([point.omega_rad_s for point in peak_points]),
            [point.response_value for point in peak_points],
            marker="^",
            s=80,
            color="tab:red",
            label="Amplification points",
            zorder=4,
        )
    if trough_points:
        ax.scatter(
            _omega_to_mrad([point.omega_rad_s for point in trough_points]),
            [point.response_value for point in trough_points],
            marker="v",
            s=80,
            color="tab:green",
            label="Attenuation points",
            zorder=4,
        )

    if omegas[0] <= theoretical_omega <= omegas[-1]:
        ax.axvline(
            theoretical_omega / 1e6,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Theoretical uncoupled resonance",
        )

    ax.set_title(f"Response metric vs angular frequency ({metric})")
    ax.set_xlabel("Angular frequency omega [Mrad/s]")
    ax.set_ylabel(_metric_label(metric))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_metric_vs_omega(
    records: Sequence[FrequencySolveRecord],
    extrema: Sequence[ExtremumRecord],
    *,
    solver: str,
    metric: Literal["iterations", "residual"],
    theoretical_omega: float,
    output_path: Path,
) -> None:
    metric_getter = (
        (lambda item: float(item.n_iter))
        if metric == "iterations"
        else (lambda item: float(item.residual))
    )
    ylabel = _metric_label(metric)
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = _color_cycle()

    preconditioners = sorted({record.preconditioner for record in records if record.solver == solver})
    for idx, preconditioner in enumerate(preconditioners):
        series = sorted(
            (
                record
                for record in records
                if record.solver == solver and record.preconditioner == preconditioner
            ),
            key=lambda item: item.omega_rad_s,
        )
        if not series:
            continue
        xvals = _omega_to_mrad([item.omega_rad_s for item in series])
        yvals = [metric_getter(item) for item in series]
        ax.plot(
            xvals,
            yvals,
            marker="o",
            markersize=2.5,
            linewidth=1.0,
            color=colors[idx % len(colors)],
            label=preconditioner,
        )

    for point in extrema:
        ax.axvline(
            point.omega_rad_s / 1e6,
            color="tab:red" if point.kind == "peak" else "tab:green",
            alpha=0.12,
            linewidth=2.0,
        )

    if theoretical_omega > 0.0:
        ax.axvline(theoretical_omega / 1e6, color="black", linestyle="--", linewidth=1.0)

    if metric == "residual":
        finite_positive = [
            record.residual
            for record in records
            if record.solver == solver and np.isfinite(record.residual) and record.residual > 0.0
        ]
        if finite_positive:
            ax.set_yscale("log")

    ax.set_title(f"{ylabel} vs angular frequency | solver={solver}")
    ax.set_xlabel("Angular frequency omega [Mrad/s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=min(3, max(1, len(preconditioners))))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_local_metric_window(
    records: Sequence[FrequencySolveRecord],
    point: ExtremumRecord,
    *,
    omegas: Sequence[float],
    solver: str,
    metric: Literal["iterations", "residual"],
    neighborhood_radius: int,
    output_path: Path,
) -> None:
    metric_getter = (
        (lambda item: float(item.n_iter))
        if metric == "iterations"
        else (lambda item: float(item.residual))
    )
    ylabel = _metric_label(metric)
    idxs = _neighbor_indices(point.index, neighborhood_radius, len(omegas))
    omega_subset = {float(omegas[idx]) for idx in idxs}
    colors = _color_cycle()

    fig, ax = plt.subplots(figsize=(10, 4))
    preconditioners = sorted({record.preconditioner for record in records if record.solver == solver})
    for idx, preconditioner in enumerate(preconditioners):
        series = sorted(
            (
                record
                for record in records
                if record.solver == solver
                and record.preconditioner == preconditioner
                and record.omega_rad_s in omega_subset
            ),
            key=lambda item: item.omega_rad_s,
        )
        if not series:
            continue
        xvals = _omega_to_mrad([item.omega_rad_s for item in series])
        yvals = [metric_getter(item) for item in series]
        ax.plot(
            xvals,
            yvals,
            marker="o",
            linewidth=1.0,
            markersize=3.0,
            color=colors[idx % len(colors)],
            label=preconditioner,
        )

    ax.axvline(point.omega_rad_s / 1e6, color="black", linestyle="--", linewidth=1.0)
    if metric == "residual":
        finite_values = [
            record.residual
            for record in records
            if record.solver == solver
            and record.omega_rad_s in omega_subset
            and np.isfinite(record.residual)
            and record.residual > 0.0
        ]
        if finite_values:
            ax.set_yscale("log")

    ax.set_title(
        f"Local {ylabel.lower()} window | {point.kind} #{point.rank} | solver={solver}"
    )
    ax.set_xlabel("Angular frequency omega [Mrad/s]")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=min(3, max(1, len(preconditioners))))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_histogram_for_point(
    records: Sequence[FrequencySolveRecord],
    point: ExtremumRecord,
    *,
    omegas: Sequence[float],
    solver: str,
    metric: Literal["iterations", "residual"],
    neighborhood_radius: int,
    output_path: Path,
) -> None:
    idxs = _neighbor_indices(point.index, neighborhood_radius, len(omegas))
    omega_subset = {float(omegas[idx]) for idx in idxs}
    colors = _color_cycle()

    fig, ax = plt.subplots(figsize=(9, 4))
    preconditioners = sorted({record.preconditioner for record in records if record.solver == solver})

    all_values: list[float] = []
    grouped_values: list[tuple[str, list[float]]] = []
    for preconditioner in preconditioners:
        values = [
            float(record.n_iter if metric == "iterations" else record.residual)
            for record in records
            if record.solver == solver
            and record.preconditioner == preconditioner
            and record.omega_rad_s in omega_subset
        ]
        if not values:
            continue
        grouped_values.append((preconditioner, values))
        all_values.extend(values)

    if not grouped_values:
        plt.close(fig)
        return

    if metric == "iterations":
        lo = int(np.floor(min(all_values)))
        hi = int(np.ceil(max(all_values)))
        bins = np.arange(lo - 0.5, hi + 1.5, 1.0)
        if bins.size < 2:
            bins = np.array([lo - 0.5, hi + 0.5], dtype=np.float64)
    else:
        finite_positive = np.asarray(
            [value for value in all_values if np.isfinite(value) and value > 0.0],
            dtype=np.float64,
        )
        if finite_positive.size == 0:
            plt.close(fig)
            return
        bins = np.logspace(
            np.log10(np.min(finite_positive)),
            np.log10(np.max(finite_positive)),
            num=min(24, max(8, finite_positive.size)),
        )
        ax.set_xscale("log")

    for idx, (preconditioner, values) in enumerate(grouped_values):
        color = colors[idx % len(colors)]
        ax.hist(
            values,
            bins=bins,
            alpha=0.35,
            color=color,
            edgecolor=color,
            label=preconditioner,
        )

    ax.set_title(f"{_metric_label(metric)} histogram | {point.kind} #{point.rank} | solver={solver}")
    ax.set_xlabel(_metric_label(metric))
    ax.set_ylabel("Count in neighborhood")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=min(3, max(1, len(grouped_values))))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_summary_report(
    path: Path,
    *,
    dataset: CompressedDataset,
    omegas: Sequence[float],
    response_metric: ResponseMetric,
    extrema: Sequence[ExtremumRecord],
    skipped: Sequence[str],
    neighborhood_summaries: Sequence[NeighborhoodSummaryRecord],
) -> None:
    theoretical_omega, theoretical_frequency_hz = theoretical_resonance(dataset)
    by_point: dict[tuple[str, int], list[NeighborhoodSummaryRecord]] = {}
    for row in neighborhood_summaries:
        by_point.setdefault((row.point_kind, row.point_rank), []).append(row)

    lines: list[str] = []
    lines.append("Resonance region analysis")
    lines.append("")
    lines.append(f"Dataset: {dataset.data_dir}")
    lines.append(f"Matrix size N: {dataset.n}")
    lines.append(f"Scanned angular frequencies: {len(omegas)}")
    lines.append(f"Angular frequency window: [{min(omegas):.6e}, {max(omegas):.6e}] rad/s")
    lines.append(f"Theoretical uncoupled omega0: {theoretical_omega:.6e} rad/s")
    lines.append(f"Theoretical uncoupled f0: {theoretical_frequency_hz:.6e} Hz")
    lines.append(f"Response metric for extrema: {response_metric}")
    lines.append("")
    lines.append("Selected amplification / attenuation points:")
    for point in extrema:
        delta_to_theory = point.omega_rad_s - theoretical_omega
        lines.append(
            f"  - {point.kind} #{point.rank}: omega={point.omega_rad_s:.6e}, "
            f"value={point.response_value:.6e}, delta_to_omega0={delta_to_theory:.6e}"
        )
        lines.append(
            f"    neighborhood=[{point.neighborhood_start_omega_rad_s:.6e}, "
            f"{point.neighborhood_end_omega_rad_s:.6e}]"
        )
        point_rows = by_point.get((point.kind, point.rank), [])
        if point_rows:
            ranked_iter = sorted(
                point_rows,
                key=lambda row: (
                    row.center_iterations,
                    row.center_minus_neighborhood_iterations
                    if np.isfinite(row.center_minus_neighborhood_iterations)
                    else -np.inf,
                ),
                reverse=True,
            )[:3]
            ranked_res = sorted(
                point_rows,
                key=lambda row: (
                    row.center_to_neighborhood_residual_ratio
                    if np.isfinite(row.center_to_neighborhood_residual_ratio)
                    else -np.inf
                ),
                reverse=True,
            )[:3]
            lines.append("    highest iteration rows:")
            for row in ranked_iter:
                lines.append(
                    f"      * {row.preconditioner} + {row.solver}: center_iter={row.center_iterations}, "
                    f"neighbor_iter_mean={row.neighborhood_iterations_mean:.3f}"
                )
            lines.append("    strongest residual deviations:")
            for row in ranked_res:
                lines.append(
                    f"      * {row.preconditioner} + {row.solver}: center_res={row.center_residual:.6e}, "
                    f"neighbor_res_mean={row.neighborhood_residual_mean:.6e}, "
                    f"ratio={row.center_to_neighborhood_residual_ratio:.3f}"
                )
    if skipped:
        lines.append("")
        lines.append("Skipped:")
        for item in skipped:
            lines.append(f"  - {item}")

    path.write_text("\n".join(lines) + "\n")


def save_plots(
    output_dir: Path,
    *,
    response: Sequence[ResponseRecord],
    extrema: Sequence[ExtremumRecord],
    solve_records: Sequence[FrequencySolveRecord],
    omegas: Sequence[float],
    response_metric: ResponseMetric,
    neighborhood_radius: int,
    theoretical_omega: float,
) -> None:
    plot_response_vs_omega(
        response,
        extrema,
        metric=response_metric,
        theoretical_omega=theoretical_omega,
        output_path=output_dir / "response_vs_omega.png",
    )

    solvers = sorted({record.solver for record in solve_records})
    for solver in solvers:
        plot_metric_vs_omega(
            solve_records,
            extrema,
            solver=solver,
            metric="iterations",
            theoretical_omega=theoretical_omega,
            output_path=output_dir / f"iterations_vs_omega__{solver}.png",
        )
        plot_metric_vs_omega(
            solve_records,
            extrema,
            solver=solver,
            metric="residual",
            theoretical_omega=theoretical_omega,
            output_path=output_dir / f"residual_vs_omega__{solver}.png",
        )

        for point in extrema:
            point_tag = f"{point.kind}_{point.rank:02d}"
            plot_local_metric_window(
                solve_records,
                point,
                omegas=omegas,
                solver=solver,
                metric="iterations",
                neighborhood_radius=neighborhood_radius,
                output_path=output_dir / f"local_iterations__{point_tag}__{solver}.png",
            )
            plot_local_metric_window(
                solve_records,
                point,
                omegas=omegas,
                solver=solver,
                metric="residual",
                neighborhood_radius=neighborhood_radius,
                output_path=output_dir / f"local_residual__{point_tag}__{solver}.png",
            )
            plot_histogram_for_point(
                solve_records,
                point,
                omegas=omegas,
                solver=solver,
                metric="iterations",
                neighborhood_radius=neighborhood_radius,
                output_path=output_dir / f"hist_iterations__{point_tag}__{solver}.png",
            )
            plot_histogram_for_point(
                solve_records,
                point,
                omegas=omegas,
                solver=solver,
                metric="residual",
                neighborhood_radius=neighborhood_radius,
                output_path=output_dir / f"hist_residual__{point_tag}__{solver}.png",
            )


def main(
    *,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    res_min: float = 38e6,
    res_max: float = 42e6,
    freq_limit: int | None = None,
    response_metric: ResponseMetric = "relative_gain",
    point_count: int = 3,
    min_separation: int = 8,
    neighborhood_radius: int = 8,
    tol: float = 1e-6,
    maxiter: int = 200,
    restart: int = 40,
    ignore_size_limits: bool = False,
    include_ridge: bool = True,
    ridge_reg: float = 1e-3,
    ridge_target: TargetKind = "pinv",
    ridge_train_limit: int | None = None,
    ridge_max_n: int = 64,
    preconditioners: Sequence[str] | None = None,
    solvers: Sequence[str] | None = None,
    output_dir: Path | None = None,
    make_plots: bool = True,
    verbose: bool = True,
) -> ResonanceAnalysisResult:
    dataset = load_compressed_dataset(data_dir)
    omegas = select_analysis_omegas(
        dataset,
        res_min=res_min,
        res_max=res_max,
        limit=freq_limit,
    )
    if omegas.size == 0:
        raise ValueError("No angular frequencies were selected for the requested window")

    if output_dir is None:
        output_dir = TESTS_DIR / "output_resonance_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(
            f"Dataset: {dataset.data_dir} | N={dataset.n} | omegas={len(omegas)} "
            f"| window=[{res_min:.6e}, {res_max:.6e}] rad/s"
        )

    response = scan_frequency_response(dataset, omegas)
    extrema = find_response_extrema(
        response,
        metric=response_metric,
        count_per_kind=point_count,
        min_separation=min_separation,
        neighborhood_radius=neighborhood_radius,
    )
    solve_records, skipped = run_frequency_solver_benchmark(
        dataset,
        omegas,
        response,
        tol=tol,
        maxiter=maxiter,
        restart=restart,
        ignore_size_limits=ignore_size_limits,
        include_ridge=include_ridge,
        ridge_reg=ridge_reg,
        ridge_target=ridge_target,
        ridge_train_limit=ridge_train_limit,
        ridge_max_n=ridge_max_n,
        preconditioner_names=preconditioners,
        solver_names=solvers,
        verbose=verbose,
    )
    neighborhood_summaries = build_neighborhood_summaries(
        solve_records,
        extrema,
        omegas=omegas,
        neighborhood_radius=neighborhood_radius,
    )

    write_csv(output_dir / "response_scan.csv", response)
    write_csv(output_dir / "selected_extrema.csv", extrema)
    write_csv(output_dir / "frequency_solve_records.csv", solve_records)
    write_csv(output_dir / "neighborhood_summaries.csv", neighborhood_summaries)

    theoretical_omega, _ = theoretical_resonance(dataset)
    write_summary_report(
        output_dir / "summary.txt",
        dataset=dataset,
        omegas=omegas,
        response_metric=response_metric,
        extrema=extrema,
        skipped=skipped,
        neighborhood_summaries=neighborhood_summaries,
    )

    if make_plots:
        save_plots(
            output_dir,
            response=response,
            extrema=extrema,
            solve_records=solve_records,
            omegas=omegas,
            response_metric=response_metric,
            neighborhood_radius=neighborhood_radius,
            theoretical_omega=theoretical_omega,
        )

    if verbose:
        print(f"Saved analysis outputs to: {output_dir}")
        for point in extrema:
            print(
                f"[{point.kind} #{point.rank}] omega={point.omega_rad_s:.6e}, "
                f"value={point.response_value:.6e}"
            )
        if skipped:
            print("Skipped:")
            for item in skipped:
                print(f"  - {item}")

    return ResonanceAnalysisResult(
        dataset=dataset,
        omegas=tuple(float(omega) for omega in omegas),
        response=tuple(response),
        extrema=tuple(extrema),
        solve_records=tuple(solve_records),
        neighborhood_summaries=tuple(neighborhood_summaries),
        skipped=tuple(skipped),
        output_dir=output_dir,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze resonance-region behavior on compressed real impedance data.",
    )
    parser.add_argument(
        "--dataset",
        choices=("small", "large"),
        default=None,
        help="Named dataset alias from tests/data.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory with unique_M.npy, pair_map.npy, flux.npy and params.json.",
    )
    parser.add_argument("--res-min", type=float, default=38e6, help="Lower omega bound [rad/s].")
    parser.add_argument("--res-max", type=float, default=42e6, help="Upper omega bound [rad/s].")
    parser.add_argument(
        "--freq-limit",
        type=int,
        default=None,
        help="Optional uniform subsampling count inside the resonance window.",
    )
    parser.add_argument(
        "--response-metric",
        choices=("relative_gain", "response_norm", "max_current_abs"),
        default="relative_gain",
        help="Metric used to locate amplification and attenuation points.",
    )
    parser.add_argument(
        "--point-count",
        type=int,
        default=3,
        help="Number of amplification points and attenuation points to keep.",
    )
    parser.add_argument(
        "--min-separation",
        type=int,
        default=8,
        help="Minimum distance in samples between selected extrema.",
    )
    parser.add_argument(
        "--neighborhood-radius",
        type=int,
        default=8,
        help="Number of samples to include on each side of a selected point.",
    )
    parser.add_argument("--tol", type=float, default=1e-6, help="Relative solver tolerance.")
    parser.add_argument("--maxiter", type=int, default=200, help="Maximum solver iterations.")
    parser.add_argument("--restart", type=int, default=40, help="GMRES restart parameter.")
    parser.add_argument(
        "--preconditioners",
        nargs="*",
        default=None,
        help="Optional list of preconditioner names to keep.",
    )
    parser.add_argument(
        "--solvers",
        nargs="*",
        default=None,
        help="Optional list of solver names to keep.",
    )
    parser.add_argument(
        "--no-ridge",
        action="store_true",
        help="Disable RidgeApprox.",
    )
    parser.add_argument(
        "--ridge-reg",
        type=float,
        default=1e-3,
        help="Regularization used by RidgeApprox.",
    )
    parser.add_argument(
        "--ridge-target",
        choices=("inverse", "pinv", "diagonal_inverse"),
        default="pinv",
        help="Target operator for RidgeApprox training.",
    )
    parser.add_argument(
        "--ridge-train-limit",
        type=int,
        default=None,
        help="Optional number of angular frequencies used to train RidgeApprox.",
    )
    parser.add_argument(
        "--ridge-max-n",
        type=int,
        default=64,
        help="Maximum matrix size for RidgeApprox training.",
    )
    parser.add_argument(
        "--ignore-size-limits",
        action="store_true",
        help="Force expensive preconditioners even if their nominal max_n limit is exceeded.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where CSV files, plots and the text summary are saved.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not save plots, only CSV files and summary.txt.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress printing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_dir = resolve_data_dir(dataset=args.dataset, data_dir=args.data_dir)
    main(
        data_dir=data_dir,
        res_min=args.res_min,
        res_max=args.res_max,
        freq_limit=args.freq_limit,
        response_metric=args.response_metric,
        point_count=args.point_count,
        min_separation=args.min_separation,
        neighborhood_radius=args.neighborhood_radius,
        tol=args.tol,
        maxiter=args.maxiter,
        restart=args.restart,
        ignore_size_limits=args.ignore_size_limits,
        include_ridge=not args.no_ridge,
        ridge_reg=args.ridge_reg,
        ridge_target=args.ridge_target,
        ridge_train_limit=args.ridge_train_limit,
        ridge_max_n=args.ridge_max_n,
        preconditioners=args.preconditioners,
        solvers=args.solvers,
        output_dir=args.output_dir,
        make_plots=not args.no_plots,
        verbose=not args.quiet,
    )
