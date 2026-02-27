from __future__ import annotations

from typing import Iterable, Optional, Sequence
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from .experiments import AggregatedRecord, ExperimentRecord, KappaTimingRecord


def apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "figure.figsize": (10, 5),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "legend.frameon": False,
        }
    )


def _filter_agg(
    records: Iterable[AggregatedRecord],
    matrix_family: Optional[str],
    solver: Optional[str],
) -> list[AggregatedRecord]:
    out: list[AggregatedRecord] = []
    for rec in records:
        if matrix_family is not None and rec.matrix_family != matrix_family:
            continue
        if solver is not None and rec.solver != solver:
            continue
        out.append(rec)
    return out


def _positive(values: Sequence[float]) -> bool:
    arr = np.asarray(values, dtype=np.float64)
    return bool(arr.size > 0 and np.all(np.isfinite(arr)) and np.all(arr > 0.0))


def _ordered_uniform_sample(
    xs: Sequence[float],
    ys: Sequence[float],
    max_points: Optional[int],
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.size == 0:
        return x, y
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if max_points is None or x.size <= max_points:
        return x, y
    idx = np.linspace(0, x.size - 1, num=max_points, dtype=int)
    return x[idx], y[idx]


def plot_delta_kappa_vs_n(
    records: Iterable[AggregatedRecord],
    matrix_family: str,
    solver: str,
    metric: str = "ratio",
    preconditioners: Optional[Sequence[str]] = None,
    *,
    log_y: Optional[bool] = None,
    ax: Optional[Axes] = None,
) -> tuple[plt.Figure, Axes]:
    metric_key = "ratio" if metric == "ratio" else "delta"
    recs = _filter_agg(records, matrix_family, solver)
    by_precond: dict[str, list[AggregatedRecord]] = {}
    for rec in recs:
        if preconditioners is not None and rec.preconditioner not in preconditioners:
            continue
        by_precond.setdefault(rec.preconditioner, []).append(rec)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    all_y: list[float] = []
    names = preconditioners if preconditioners is not None else sorted(by_precond)
    for precond in names:
        series = by_precond.get(precond, [])
        if not series:
            continue
        series = sorted(series, key=lambda r: r.n)
        xs = [r.n for r in series]
        if metric_key == "ratio":
            ys = [r.ratio_kappa_mean for r in series]
            yerr = [r.ratio_kappa_std for r in series]
            ylabel = "kappa_pre / kappa (signed)"
        else:
            ys = [r.delta_kappa_mean for r in series]
            yerr = [r.delta_kappa_std for r in series]
            ylabel = "kappa_pre - kappa (signed)"
        all_y.extend(ys)
        ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=precond)

    if log_y is None:
        use_log_y = metric_key == "ratio" and _positive(all_y)
    else:
        use_log_y = log_y and _positive(all_y)

    all_x = sorted({r.n for r in recs})
    if all_x:
        ax.set_xticks(all_x)
    ax.set_title(f"Condition metric vs size | {matrix_family} | {solver}")
    ax.set_xlabel("Matrix size n")
    ax.set_ylabel(ylabel if by_precond else "Metric")
    ax.set_yscale("log" if use_log_y else "linear")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_iterations_vs_n(
    records: Iterable[AggregatedRecord],
    matrix_family: str,
    solver: str,
    preconditioners: Optional[Sequence[str]] = None,
    *,
    ax: Optional[Axes] = None,
) -> tuple[plt.Figure, Axes]:
    recs = _filter_agg(records, matrix_family, solver)
    by_precond: dict[str, list[AggregatedRecord]] = {}
    for rec in recs:
        if preconditioners is not None and rec.preconditioner not in preconditioners:
            continue
        by_precond.setdefault(rec.preconditioner, []).append(rec)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    names = preconditioners if preconditioners is not None else sorted(by_precond)
    for precond in names:
        series = by_precond.get(precond, [])
        if not series:
            continue
        series = sorted(series, key=lambda r: r.n)
        xs = [r.n for r in series]
        ys = [r.iterations_mean for r in series]
        yerr = [r.iterations_std for r in series]
        ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=precond)

    all_x = sorted({r.n for r in recs})
    if all_x:
        ax.set_xticks(all_x)
    ax.set_title(f"Iterations vs size | {matrix_family} | {solver}")
    ax.set_xlabel("Matrix size n")
    ax.set_ylabel("Mean iterations")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_delta_kappa_vs_kappa(
    records: Iterable[ExperimentRecord],
    matrix_family: str,
    solver: str,
    metric: str = "ratio",
    preconditioners: Optional[Sequence[str]] = None,
    *,
    x_log: bool = False,
    y_log: Optional[bool] = None,
    max_points_per_preconditioner: Optional[int] = None,
    ax: Optional[Axes] = None,
) -> tuple[plt.Figure, Axes]:
    metric_key = "ratio" if metric == "ratio" else "delta"
    xs: dict[str, list[float]] = {}
    ys: dict[str, list[float]] = {}

    for rec in records:
        if rec.matrix_family != matrix_family or rec.solver != solver:
            continue
        if preconditioners is not None and rec.preconditioner not in preconditioners:
            continue
        xs.setdefault(rec.preconditioner, []).append(rec.kappa)
        ys.setdefault(rec.preconditioner, []).append(
            rec.ratio_kappa if metric_key == "ratio" else rec.delta_kappa
        )

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    all_x: list[float] = []
    all_y: list[float] = []
    names = preconditioners if preconditioners is not None else sorted(xs)
    for precond in names:
        x_raw = xs.get(precond, [])
        y_raw = ys.get(precond, [])
        if not x_raw:
            continue
        xvals, yvals = _ordered_uniform_sample(
            x_raw, y_raw, max_points_per_preconditioner
        )
        all_x.extend(xvals.tolist())
        all_y.extend(yvals.tolist())
        ax.scatter(xvals, yvals, alpha=0.6, label=precond)

    use_x_log = x_log and _positive(all_x)
    if y_log is None:
        use_y_log = metric_key == "ratio" and _positive(all_y)
    else:
        use_y_log = y_log and _positive(all_y)

    ax.set_title(f"Condition metric vs kappa(A) | {matrix_family} | {solver}")
    ax.set_xlabel("kappa(A) signed")
    ax.set_ylabel(
        "kappa_pre / kappa (signed)"
        if metric_key == "ratio"
        else "kappa_pre - kappa (signed)"
    )
    ax.set_xscale("log" if use_x_log else "linear")
    ax.set_yscale("log" if use_y_log else "linear")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_time_vs_n(
    records: Iterable[AggregatedRecord],
    matrix_family: str,
    solver: str,
    *,
    metric: str = "solve",
    preconditioners: Optional[Sequence[str]] = None,
    log_y: bool = False,
    ax: Optional[Axes] = None,
) -> tuple[plt.Figure, Axes]:
    metric_map = {
        "solve": ("solve_time_mean", "solve_time_std", "Solve time [s]"),
        "total": ("total_time_mean", "total_time_std", "Total pipeline time [s]"),
        "preconditioner": (
            "preconditioner_time_mean",
            "preconditioner_time_std",
            "Preconditioner build time [s]",
        ),
        "kappa": ("kappa_eval_time_mean", "kappa_eval_time_std", "Kappa eval time [s]"),
    }
    if metric not in metric_map:
        raise ValueError(f"Unknown metric: {metric!r}")

    mean_attr, std_attr, ylabel = metric_map[metric]
    recs = _filter_agg(records, matrix_family, solver)
    by_precond: dict[str, list[AggregatedRecord]] = {}
    for rec in recs:
        if preconditioners is not None and rec.preconditioner not in preconditioners:
            continue
        by_precond.setdefault(rec.preconditioner, []).append(rec)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    all_y: list[float] = []
    names = preconditioners if preconditioners is not None else sorted(by_precond)
    for precond in names:
        series = by_precond.get(precond, [])
        if not series:
            continue
        series = sorted(series, key=lambda r: r.n)
        xs = [r.n for r in series]
        ys = [float(getattr(r, mean_attr)) for r in series]
        yerr = [float(getattr(r, std_attr)) for r in series]
        all_y.extend(ys)
        ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=precond)

    if log_y and _positive(all_y):
        ax.set_yscale("log")
    all_x = sorted({r.n for r in recs})
    if all_x:
        ax.set_xticks(all_x)
    ax.set_title(f"Time vs size | {matrix_family} | {solver}")
    ax.set_xlabel("Matrix size n")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_time_vs_kappa(
    records: Iterable[KappaTimingRecord],
    matrix_family: str,
    solver: str,
    *,
    metric: str = "solve",
    preconditioners: Optional[Sequence[str]] = None,
    x_log: bool = False,
    y_log: bool = False,
    ax: Optional[Axes] = None,
) -> tuple[plt.Figure, Axes]:
    metric_map = {
        "solve": ("solve_time_mean", "solve_time_std", "Solve time [s]"),
        "total": ("total_time_mean", "total_time_std", "Total pipeline time [s]"),
    }
    if metric not in metric_map:
        raise ValueError(f"Unknown metric: {metric!r}")
    mean_attr, std_attr, ylabel = metric_map[metric]

    by_precond: dict[str, list[KappaTimingRecord]] = {}
    for rec in records:
        if rec.matrix_family != matrix_family or rec.solver != solver:
            continue
        if preconditioners is not None and rec.preconditioner not in preconditioners:
            continue
        by_precond.setdefault(rec.preconditioner, []).append(rec)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    all_x: list[float] = []
    all_y: list[float] = []
    names = preconditioners if preconditioners is not None else sorted(by_precond)
    for precond in names:
        series = by_precond.get(precond, [])
        if not series:
            continue
        series = sorted(series, key=lambda r: r.kappa_center)
        xs = [r.kappa_center for r in series]
        ys = [float(getattr(r, mean_attr)) for r in series]
        yerr = [float(getattr(r, std_attr)) for r in series]
        all_x.extend(xs)
        all_y.extend(ys)
        ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=precond)

    if x_log and _positive(all_x):
        ax.set_xscale("log")
    if y_log and _positive(all_y):
        ax.set_yscale("log")
    ax.set_title(f"Time vs kappa | {matrix_family} | {solver}")
    ax.set_xlabel("kappa(A) signed (bin center)")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    return fig, ax