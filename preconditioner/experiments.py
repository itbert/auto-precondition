from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, List, Tuple

import numpy as np
from numpy.typing import NDArray

from .matrices import MatrixFamily
from .metrics import compute_kappa_metrics, condition_number
from .preconditioners import PreconditionerFactory
from .solvers import SolverResult, SolverSpec


Array = NDArray[np.float64]


@dataclass(frozen=True)
class ExperimentConfig:
    sizes: Tuple[int, ...]
    n_samples: int
    matrix_families: Tuple[MatrixFamily, ...]
    preconditioners: Tuple[PreconditionerFactory, ...]
    solvers: Tuple[SolverSpec, ...]
    rng_seed: int = 42
    tol: float = 1e-8
    maxiter: int = 500
    signed_kappa: bool = True
    kappa_zero_tol: float = 1e-12


@dataclass(frozen=True)
class ExperimentRecord:
    matrix_family: str
    n: int
    sample: int
    kappa: float
    preconditioner: str
    kappa_pre: float
    delta_kappa: float
    ratio_kappa: float
    solver: str
    n_iter: int
    residual: float
    converged: bool
    preconditioner_time: float
    kappa_eval_time: float
    solve_time: float
    total_time: float


@dataclass(frozen=True)
class AggregatedRecord:
    matrix_family: str
    n: int
    preconditioner: str
    solver: str
    n_samples: int
    kappa_mean: float
    kappa_std: float
    kappa_pre_mean: float
    kappa_pre_std: float
    delta_kappa_mean: float
    delta_kappa_std: float
    ratio_kappa_mean: float
    ratio_kappa_std: float
    iterations_mean: float
    iterations_std: float
    residual_mean: float
    residual_std: float
    converge_rate: float
    preconditioner_time_mean: float
    preconditioner_time_std: float
    kappa_eval_time_mean: float
    kappa_eval_time_std: float
    solve_time_mean: float
    solve_time_std: float
    total_time_mean: float
    total_time_std: float


@dataclass(frozen=True)
class KappaTimingRecord:
    matrix_family: str
    preconditioner: str
    solver: str
    kappa_bin: int
    kappa_center: float
    n_samples: int
    solve_time_mean: float
    solve_time_std: float
    total_time_mean: float
    total_time_std: float


def _safe_solver_call(
    solver: SolverSpec,
    A: Array,
    b: NDArray[np.float64],
    M,
    tol: float,
    maxiter: int,
) -> SolverResult:
    try:
        return solver.solve(A, b, M, tol, maxiter)
    except Exception:
        return SolverResult(
            x=np.zeros_like(b),
            info=1,
            n_iter=maxiter,
            residual_norm=float(np.inf),
            residuals=tuple(),
        )


def run_experiments(config: ExperimentConfig) -> List[ExperimentRecord]:
    rng = np.random.default_rng(config.rng_seed)
    records: List[ExperimentRecord] = []

    for family in config.matrix_families:
        for n in config.sizes:
            for sample in range(config.n_samples):
                A = family.generator(n, rng)
                b = rng.standard_normal(n)
                kappa_base = condition_number(
                    A,
                    signed=config.signed_kappa,
                    zero_tol=config.kappa_zero_tol,
                )

                for precond_factory in config.preconditioners:
                    if not precond_factory.supports(n):
                        continue

                    t_pre0 = perf_counter()
                    precond = precond_factory.build(A)
                    preconditioner_time = perf_counter() - t_pre0

                    t_kappa0 = perf_counter()
                    A_pre = precond.apply_to_matrix(A)
                    kappa_pre = condition_number(
                        A_pre,
                        signed=config.signed_kappa,
                        zero_tol=config.kappa_zero_tol,
                    )
                    kappa_eval_time = perf_counter() - t_kappa0

                    kappa_metrics = compute_kappa_metrics(kappa_base, kappa_pre)
                    M = precond.as_linear_operator()

                    for solver in config.solvers:
                        t_solve0 = perf_counter()
                        result = _safe_solver_call(
                            solver, A, b, M, config.tol, config.maxiter
                        )
                        solve_time = perf_counter() - t_solve0
                        total_time = preconditioner_time + kappa_eval_time + solve_time
                        records.append(
                            ExperimentRecord(
                                matrix_family=family.name,
                                n=n,
                                sample=sample,
                                kappa=kappa_metrics.kappa,
                                preconditioner=precond_factory.name,
                                kappa_pre=kappa_metrics.kappa_pre,
                                delta_kappa=kappa_metrics.delta,
                                ratio_kappa=kappa_metrics.ratio,
                                solver=solver.name,
                                n_iter=result.n_iter,
                                residual=result.residual_norm,
                                converged=result.converged,
                                preconditioner_time=preconditioner_time,
                                kappa_eval_time=kappa_eval_time,
                                solve_time=solve_time,
                                total_time=total_time,
                            )
                        )

    return records


def _mean_std(values: Iterable[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(arr)), float(np.std(arr))


def aggregate_records(records: Iterable[ExperimentRecord]) -> List[AggregatedRecord]:
    groups: dict[tuple, list[ExperimentRecord]] = {}
    for rec in records:
        key = (rec.matrix_family, rec.n, rec.preconditioner, rec.solver)
        groups.setdefault(key, []).append(rec)

    aggregated: List[AggregatedRecord] = []
    for key in sorted(groups):
        family, n, precond, solver = key
        recs = groups[key]
        kappa_mean, kappa_std = _mean_std(r.kappa for r in recs)
        kappa_pre_mean, kappa_pre_std = _mean_std(r.kappa_pre for r in recs)
        delta_mean, delta_std = _mean_std(r.delta_kappa for r in recs)
        ratio_mean, ratio_std = _mean_std(r.ratio_kappa for r in recs)
        it_mean, it_std = _mean_std(r.n_iter for r in recs)
        res_mean, res_std = _mean_std(r.residual for r in recs)
        pre_t_mean, pre_t_std = _mean_std(r.preconditioner_time for r in recs)
        kappa_t_mean, kappa_t_std = _mean_std(r.kappa_eval_time for r in recs)
        solve_t_mean, solve_t_std = _mean_std(r.solve_time for r in recs)
        total_t_mean, total_t_std = _mean_std(r.total_time for r in recs)
        conv_rate = float(np.mean([r.converged for r in recs]))

        aggregated.append(
            AggregatedRecord(
                matrix_family=family,
                n=n,
                preconditioner=precond,
                solver=solver,
                n_samples=len(recs),
                kappa_mean=kappa_mean,
                kappa_std=kappa_std,
                kappa_pre_mean=kappa_pre_mean,
                kappa_pre_std=kappa_pre_std,
                delta_kappa_mean=delta_mean,
                delta_kappa_std=delta_std,
                ratio_kappa_mean=ratio_mean,
                ratio_kappa_std=ratio_std,
                iterations_mean=it_mean,
                iterations_std=it_std,
                residual_mean=res_mean,
                residual_std=res_std,
                converge_rate=conv_rate,
                preconditioner_time_mean=pre_t_mean,
                preconditioner_time_std=pre_t_std,
                kappa_eval_time_mean=kappa_t_mean,
                kappa_eval_time_std=kappa_t_std,
                solve_time_mean=solve_t_mean,
                solve_time_std=solve_t_std,
                total_time_mean=total_t_mean,
                total_time_std=total_t_std,
            )
        )

    return aggregated


def aggregate_timing_by_kappa(
    records: Iterable[ExperimentRecord],
    *,
    bins: int = 12,
) -> List[KappaTimingRecord]:
    if bins <= 0:
        raise ValueError("bins must be positive")

    grouped: dict[tuple[str, str, str], list[ExperimentRecord]] = {}
    for rec in records:
        if not np.isfinite(rec.kappa):
            continue
        key = (rec.matrix_family, rec.preconditioner, rec.solver)
        grouped.setdefault(key, []).append(rec)

    out: List[KappaTimingRecord] = []
    for key in sorted(grouped):
        family, precond, solver = key
        group = sorted(grouped[key], key=lambda r: r.kappa)
        n_bins = min(bins, len(group))
        chunks = np.array_split(np.asarray(group, dtype=object), n_bins)
        for idx, chunk_array in enumerate(chunks):
            chunk = [r for r in chunk_array.tolist() if isinstance(r, ExperimentRecord)]
            if not chunk:
                continue
            solve_mean, solve_std = _mean_std(r.solve_time for r in chunk)
            total_mean, total_std = _mean_std(r.total_time for r in chunk)
            kappa_center = float(np.mean([r.kappa for r in chunk]))
            out.append(
                KappaTimingRecord(
                    matrix_family=family,
                    preconditioner=precond,
                    solver=solver,
                    kappa_bin=idx,
                    kappa_center=kappa_center,
                    n_samples=len(chunk),
                    solve_time_mean=solve_mean,
                    solve_time_std=solve_std,
                    total_time_mean=total_mean,
                    total_time_std=total_std,
                )
            )
    return out