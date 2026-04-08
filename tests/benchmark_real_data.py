from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
PRECONDITIONER_SRC = ROOT_DIR / "preconditioner"
if str(PRECONDITIONER_SRC) not in sys.path:
    sys.path.insert(0, str(PRECONDITIONER_SRC))

from ml import RidgeInverseApproximator, TargetKind
from preconditioners import PreconditionerFactory, default_preconditioners
from solvers import SolverResult, default_solvers


DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data" / "compressed_data555"
DATASET_ALIASES = {
    "small": Path(__file__).resolve().parent / "data" / "compressed_data",
    "large": Path(__file__).resolve().parent / "data" / "compressed_data555",
}


@dataclass(frozen=True)
class CompressedDataset:
    data_dir: Path
    unique_M: np.ndarray
    pair_map: np.ndarray
    flux: np.ndarray
    R: float
    L: float
    C: float
    freqs: np.ndarray
    edge_i: np.ndarray
    edge_j: np.ndarray
    edge_idx: np.ndarray

    @property
    def n(self) -> int:
        return int(self.pair_map.shape[0])

    def selected_freqs(
        self,
        *,
        res_min: float,
        res_max: float,
        limit: int | None = None,
    ) -> np.ndarray:
        freqs = np.sort(self.freqs[(self.freqs >= res_min) & (self.freqs <= res_max)])
        if limit is not None:
            freqs = freqs[:limit]
        return freqs


@dataclass(frozen=True)
class SolveRecord:
    frequency_hz: float
    preconditioner: str
    solver: str
    info: int
    converged: bool
    n_iter: int
    residual: float
    preconditioner_time_s: float
    solve_time_s: float
    total_time_s: float


@dataclass(frozen=True)
class SummaryRecord:
    preconditioner: str
    solver: str
    n_freqs: int
    converged_count: int
    converge_rate: float
    iterations_mean: float
    iterations_min: int
    iterations_max: int
    residual_mean: float
    residual_max: float
    preconditioner_time_mean_s: float
    solve_time_mean_s: float
    total_time_mean_s: float


def load_compressed_dataset(data_dir: Path | str = DEFAULT_DATA_DIR) -> CompressedDataset:
    data_path = Path(data_dir)
    unique_M = np.load(data_path / "unique_M.npy")
    pair_map = np.load(data_path / "pair_map.npy")
    flux = np.load(data_path / "flux.npy")
    with (data_path / "params.json").open() as f:
        params = json.load(f)

    upper_i, upper_j = np.triu_indices(pair_map.shape[0], k=1)
    mask = pair_map[upper_i, upper_j] >= 0
    edge_i = upper_i[mask]
    edge_j = upper_j[mask]
    edge_idx = pair_map[edge_i, edge_j]

    return CompressedDataset(
        data_dir=data_path,
        unique_M=unique_M,
        pair_map=pair_map,
        flux=flux.astype(np.complex128, copy=False),
        R=float(params["R"]),
        L=float(params["L"]),
        C=float(params["C"]),
        freqs=np.asarray(params["freqs"], dtype=np.float64),
        edge_i=edge_i,
        edge_j=edge_j,
        edge_idx=edge_idx,
    )


def resolve_data_dir(
    *,
    dataset: str | None = None,
    data_dir: Path | str | None = None,
) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    if dataset is not None:
        return DATASET_ALIASES[dataset]
    return DEFAULT_DATA_DIR


def build_impedance_matrix(dataset: CompressedDataset, omega: float) -> np.ndarray:
    Z = np.zeros((dataset.n, dataset.n), dtype=np.complex128)
    diag = dataset.R - 1j * omega * dataset.L - 1.0 / (1j * omega * dataset.C)
    np.fill_diagonal(Z, diag)
    offdiag = -1j * omega * dataset.unique_M[dataset.edge_idx]
    Z[dataset.edge_i, dataset.edge_j] = offdiag
    Z[dataset.edge_j, dataset.edge_i] = offdiag
    return Z


def build_rhs(dataset: CompressedDataset, omega: float) -> np.ndarray:
    return -1j * omega * dataset.flux


def _select_training_freqs(freqs: np.ndarray, limit: int | None) -> np.ndarray:
    if limit is None or limit >= len(freqs):
        return freqs
    indices = np.linspace(0, len(freqs) - 1, num=limit, dtype=int)
    return freqs[np.unique(indices)]


def _build_inverse_target(A: np.ndarray, target: TargetKind) -> np.ndarray:
    if target == "inverse":
        return np.linalg.inv(A)
    if target == "pinv":
        return np.linalg.pinv(A)
    if target == "diagonal_inverse":
        d = np.diag(A)
        inv_d = np.where(np.abs(d) > 1e-12, 1.0 / d, 1.0)
        return np.diag(inv_d)
    raise ValueError(f"Unknown ridge target: {target!r}")


def build_ridge_preconditioner_factory(
    dataset: CompressedDataset,
    freqs: np.ndarray,
    *,
    ridge_reg: float = 1e-3,
    ridge_target: TargetKind = "pinv",
    ridge_train_limit: int | None = None,
    ridge_max_n: int = 64,
    verbose: bool = True,
) -> tuple[PreconditionerFactory | None, str]:
    if dataset.n > ridge_max_n:
        return (
            None,
            f"RidgeApprox: skipped for N={dataset.n} "
            f"(ridge_max_n={ridge_max_n}; dense ridge model is too large)",
        )

    train_freqs = _select_training_freqs(freqs, ridge_train_limit)
    if len(train_freqs) == 0:
        return None, "RidgeApprox: skipped because no frequencies were selected"

    if verbose:
        print(
            f"[RidgeApprox] training on {len(train_freqs)} matrices "
            f"(target={ridge_target}, ridge={ridge_reg:.1e})..."
        )

    t0 = perf_counter()
    train_matrices = [build_impedance_matrix(dataset, float(omega)) for omega in train_freqs]
    train_targets = [_build_inverse_target(A, ridge_target) for A in train_matrices]
    model = RidgeInverseApproximator(matrix_size=dataset.n, ridge=ridge_reg).fit(
        train_matrices, train_targets
    )
    elapsed = perf_counter() - t0

    if verbose:
        print(f"[RidgeApprox] trained in {elapsed:.3f} s")

    return (
        PreconditionerFactory(
            "RidgeApprox",
            lambda A, model=model: model.as_preconditioner(A, name="RidgeApprox"),
            max_n=ridge_max_n,
            notes="Learned inverse approximator on real data",
        ),
        (
            f"RidgeApprox: trained on {len(train_freqs)} matrices; "
            f"offline training time {elapsed:.3f} s is not included in per-matrix timings"
        ),
    )


def summarize_records(records: Iterable[SolveRecord]) -> list[SummaryRecord]:
    grouped: dict[tuple[str, str], list[SolveRecord]] = {}
    for record in records:
        key = (record.preconditioner, record.solver)
        grouped.setdefault(key, []).append(record)

    summary: list[SummaryRecord] = []
    for preconditioner, solver in sorted(grouped):
        group = grouped[(preconditioner, solver)]
        iterations = np.asarray([r.n_iter for r in group], dtype=np.int64)
        residuals = np.asarray([r.residual for r in group], dtype=np.float64)
        pre_times = np.asarray([r.preconditioner_time_s for r in group], dtype=np.float64)
        solve_times = np.asarray([r.solve_time_s for r in group], dtype=np.float64)
        total_times = np.asarray([r.total_time_s for r in group], dtype=np.float64)
        converged = np.asarray([r.converged for r in group], dtype=np.float64)

        summary.append(
            SummaryRecord(
                preconditioner=preconditioner,
                solver=solver,
                n_freqs=len(group),
                converged_count=int(np.sum(converged)),
                converge_rate=float(np.mean(converged)),
                iterations_mean=float(np.mean(iterations)),
                iterations_min=int(np.min(iterations)),
                iterations_max=int(np.max(iterations)),
                residual_mean=float(np.mean(residuals)),
                residual_max=float(np.max(residuals)),
                preconditioner_time_mean_s=float(np.mean(pre_times)),
                solve_time_mean_s=float(np.mean(solve_times)),
                total_time_mean_s=float(np.mean(total_times)),
            )
        )
    return summary


def run_real_data_benchmark(
    *,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    res_min: float = 38e6,
    res_max: float = 42e6,
    freq_limit: int | None = None,
    tol: float = 1e-6,
    maxiter: int = 200,
    restart: int = 40,
    ignore_size_limits: bool = False,
    include_ridge: bool = True,
    ridge_reg: float = 1e-3,
    ridge_target: TargetKind = "pinv",
    ridge_train_limit: int | None = None,
    ridge_max_n: int = 64,
    verbose: bool = True,
) -> tuple[list[SummaryRecord], list[SolveRecord], list[str]]:
    dataset = load_compressed_dataset(data_dir)
    freqs = dataset.selected_freqs(res_min=res_min, res_max=res_max, limit=freq_limit)
    preconditioners = list(default_preconditioners())
    solvers = tuple(default_solvers(restart=restart))
    skipped: list[str] = []
    records: list[SolveRecord] = []

    if verbose:
        print(
            f"Dataset: {dataset.data_dir} | N={dataset.n} | freqs={len(freqs)} "
            f"| window=[{res_min:.0f}, {res_max:.0f}] Hz"
        )

    if include_ridge:
        ridge_factory, ridge_message = build_ridge_preconditioner_factory(
            dataset,
            freqs,
            ridge_reg=ridge_reg,
            ridge_target=ridge_target,
            ridge_train_limit=ridge_train_limit,
            ridge_max_n=ridge_max_n,
            verbose=verbose,
        )
        if ridge_factory is None:
            skipped.append(ridge_message)
        else:
            preconditioners.append(ridge_factory)
            if verbose:
                print(ridge_message)

    for preconditioner in preconditioners:
        if not ignore_size_limits and not preconditioner.supports(dataset.n):
            skipped.append(
                f"{preconditioner.name}: skipped for N={dataset.n} "
                f"(limit={preconditioner.max_n})"
            )
            continue

        if verbose:
            print(f"[{preconditioner.name}] running {len(freqs)} frequencies...")

        for omega in freqs:
            Z = build_impedance_matrix(dataset, float(omega))
            v = build_rhs(dataset, float(omega))

            t_pre0 = perf_counter()
            try:
                built = preconditioner.build(Z)
            except Exception as exc:
                pre_time = perf_counter() - t_pre0
                for solver in solvers:
                    records.append(
                        SolveRecord(
                            frequency_hz=float(omega),
                            preconditioner=preconditioner.name,
                            solver=solver.name,
                            info=-1,
                            converged=False,
                            n_iter=0,
                            residual=float("inf"),
                            preconditioner_time_s=pre_time,
                            solve_time_s=0.0,
                            total_time_s=pre_time,
                        )
                    )
                skipped.append(
                    f"{preconditioner.name}: build failed at {omega:.0f} Hz "
                    f"with {type(exc).__name__}: {exc}"
                )
                continue
            pre_time = perf_counter() - t_pre0
            linear_operator = built.as_linear_operator()

            for solver in solvers:
                t_solve0 = perf_counter()
                try:
                    result: SolverResult = solver.solve(Z, v, linear_operator, tol, maxiter)
                    info = int(result.info)
                    converged = bool(result.converged)
                    n_iter = int(result.n_iter)
                    residual = float(np.linalg.norm(v - Z @ result.x) / np.linalg.norm(v))
                except Exception:
                    info = maxiter
                    converged = False
                    n_iter = maxiter
                    residual = float("inf")
                solve_time = perf_counter() - t_solve0

                records.append(
                    SolveRecord(
                        frequency_hz=float(omega),
                        preconditioner=preconditioner.name,
                        solver=solver.name,
                        info=info,
                        converged=converged,
                        n_iter=n_iter,
                        residual=residual,
                        preconditioner_time_s=pre_time,
                        solve_time_s=solve_time,
                        total_time_s=pre_time + solve_time,
                    )
                )

    return summarize_records(records), records, skipped


def _format_float(value: float) -> str:
    if np.isfinite(value):
        if abs(value) >= 1e-2:
            return f"{value:.4f}"
        return f"{value:.3e}"
    return "inf"


def print_summary(summary: Iterable[SummaryRecord], *, skipped: Iterable[str] = ()) -> None:
    rows = list(summary)
    if not rows:
        print("No benchmark rows were produced.")
        return

    header = (
        f"{'Preconditioner':<14} {'Solver':<9} {'Conv':>9} {'Iter mean':>10} "
        f"{'Iter min':>8} {'Iter max':>8} {'Residual mean':>14} "
        f"{'Total time mean, s':>18}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        conv_text = f"{row.converged_count}/{row.n_freqs}"
        print(
            f"{row.preconditioner:<14} {row.solver:<9} {conv_text:>9} "
            f"{row.iterations_mean:>10.2f} {row.iterations_min:>8d} "
            f"{row.iterations_max:>8d} {_format_float(row.residual_mean):>14} "
            f"{_format_float(row.total_time_mean_s):>18}"
        )

    skipped_items = list(skipped)
    if skipped_items:
        print("\nSkipped:")
        for item in skipped_items:
            print(f"  - {item}")


def print_series(records: Iterable[SolveRecord]) -> None:
    grouped: dict[tuple[str, str], list[SolveRecord]] = {}
    for record in records:
        grouped.setdefault((record.preconditioner, record.solver), []).append(record)

    for preconditioner, solver in sorted(grouped):
        group = grouped[(preconditioner, solver)]
        iterations = [record.n_iter for record in group]
        infos = [record.info for record in group]
        residuals = [record.residual for record in group]
        print(f"\n{preconditioner} + {solver}")
        print(f"  iterations = {iterations}")
        print(f"  infos      = {infos}")
        print(f"  residuals  = {[float(f'{value:.6e}') for value in residuals]}")


def write_csv(path: Path, rows: Iterable[object]) -> None:
    rows_list = list(rows)
    if not rows_list:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows_list[0]).keys()))
        writer.writeheader()
        for row in rows_list:
            writer.writerow(asdict(row))


def main(
    *,
    data_dir: Path | str = DEFAULT_DATA_DIR,
    res_min: float = 38e6,
    res_max: float = 42e6,
    freq_limit: int | None = None,
    tol: float = 1e-6,
    maxiter: int = 200,
    restart: int = 40,
    ignore_size_limits: bool = False,
    include_ridge: bool = True,
    ridge_reg: float = 1e-3,
    ridge_target: TargetKind = "pinv",
    ridge_train_limit: int | None = None,
    ridge_max_n: int = 64,
    print_raw: bool = False,
    csv_dir: Path | None = None,
) -> tuple[list[SummaryRecord], list[SolveRecord]]:
    summary, records, skipped = run_real_data_benchmark(
        data_dir=data_dir,
        res_min=res_min,
        res_max=res_max,
        freq_limit=freq_limit,
        tol=tol,
        maxiter=maxiter,
        restart=restart,
        ignore_size_limits=ignore_size_limits,
        include_ridge=include_ridge,
        ridge_reg=ridge_reg,
        ridge_target=ridge_target,
        ridge_train_limit=ridge_train_limit,
        ridge_max_n=ridge_max_n,
    )
    print_summary(summary, skipped=skipped)

    if print_raw:
        print_series(records)

    if csv_dir is not None:
        write_csv(csv_dir / "benchmark_real_data_summary.csv", summary)
        write_csv(csv_dir / "benchmark_real_data_raw.csv", records)

    return summary, records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark real compressed impedance data.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_ALIASES),
        default=None,
        help="Named dataset alias: 'small' for compressed_data, 'large' for compressed_data555",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory with unique_M.npy, pair_map.npy, flux.npy and params.json",
    )
    parser.add_argument("--res-min", type=float, default=38e6, help="Lower frequency bound, Hz")
    parser.add_argument("--res-max", type=float, default=42e6, help="Upper frequency bound, Hz")
    parser.add_argument("--freq-limit", type=int, default=None, help="Optional frequency limit")
    parser.add_argument("--tol", type=float, default=1e-6, help="Relative solver tolerance")
    parser.add_argument("--maxiter", type=int, default=200, help="Maximum solver iterations")
    parser.add_argument("--restart", type=int, default=40, help="GMRES restart parameter")
    parser.add_argument(
        "--no-ridge",
        action="store_true",
        help="Disable RidgeApprox preconditioner from preconditioner/ml.py",
    )
    parser.add_argument(
        "--ridge-reg",
        type=float,
        default=1e-3,
        help="Regularization used by RidgeApprox",
    )
    parser.add_argument(
        "--ridge-target",
        choices=("inverse", "pinv", "diagonal_inverse"),
        default="pinv",
        help="Target operator for RidgeApprox training",
    )
    parser.add_argument(
        "--ridge-train-limit",
        type=int,
        default=None,
        help="Optional number of frequencies used to train RidgeApprox",
    )
    parser.add_argument(
        "--ridge-max-n",
        type=int,
        default=64,
        help="Maximum matrix size for RidgeApprox training",
    )
    parser.add_argument(
        "--ignore-size-limits",
        action="store_true",
        help="Force expensive preconditioners even if their nominal max_n limit is exceeded",
    )
    parser.add_argument(
        "--print-raw",
        action="store_true",
        help="Print per-frequency iteration/info/residual series for each method",
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=None,
        help="Optional output directory for summary/raw CSV exports",
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
        tol=args.tol,
        maxiter=args.maxiter,
        restart=args.restart,
        ignore_size_limits=args.ignore_size_limits,
        include_ridge=not args.no_ridge,
        ridge_reg=args.ridge_reg,
        ridge_target=args.ridge_target,
        ridge_train_limit=args.ridge_train_limit,
        ridge_max_n=args.ridge_max_n,
        print_raw=args.print_raw,
        csv_dir=args.csv_dir,
    )
