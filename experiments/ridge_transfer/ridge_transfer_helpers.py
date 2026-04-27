from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
import tempfile
from typing import Iterable, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "auto_preconditioner_ridge_transfer_mpl"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from preconditioner import (
    MatrixFamily,
    PreconditionerWorkbenchV2,
    RidgeInverseApproximator,
    apply_plot_style,
    build_inverse_dataset,
    condition_number,
)
from preconditioner.preconditioners import diagonal_preconditioner
from preconditioner.solvers import solve_gmres


Array = NDArray[np.float64]


@dataclass(frozen=True)
class TransferExperimentConfig:
    matrix_size: int = 24
    ridge_reg: float = 1e-3
    target: str = "pinv"
    gmres_tol: float = 1e-8
    gmres_maxiter: int = 200
    gmres_restart: int = 40
    base_train_samples: int = 72
    base_val_samples: int = 24
    shifted_eval_samples: int = 24
    fine_tune_sizes: tuple[int, ...] = (0, 1, 2, 4, 6, 8, 12)
    showcase_fine_tune_size: int = 4
    basis_seed: int = 0
    train_seed: int = 0
    val_seed: int = 1
    shifted_seed: int = 2
    base_rho_range: tuple[float, float] = (0.10, 0.30)
    base_shift_range: tuple[float, float] = (0.05, 0.20)
    base_wobble_range: tuple[float, float] = (0.00, 0.06)
    shifted_rho_range: tuple[float, float] = (0.32, 0.44)
    shifted_shift_range: tuple[float, float] = (0.26, 0.36)
    shifted_wobble_range: tuple[float, float] = (0.08, 0.14)
    rhs_seed: int = 17


@dataclass(frozen=True)
class TransferRecord:
    regime: str
    split: str
    sample_index: int
    matrix_kappa: float
    relative_fro_error: float
    preconditioned_kappa: float
    gmres_iterations: int
    gmres_residual: float
    no_preconditioner_iterations: int
    diagonal_iterations: int


@dataclass(frozen=True)
class TransferSummary:
    regime: str
    split: str
    adaptation_samples: int
    n_samples: int
    matrix_kappa_mean: float
    matrix_kappa_std: float
    relative_fro_error_mean: float
    relative_fro_error_std: float
    preconditioned_kappa_mean: float
    preconditioned_kappa_std: float
    gmres_iterations_mean: float
    gmres_iterations_std: float
    gmres_residual_mean: float
    gmres_residual_std: float
    no_preconditioner_iterations_mean: float
    diagonal_iterations_mean: float


@dataclass(frozen=True)
class TransferBundle:
    config: TransferExperimentConfig
    workbench: PreconditionerWorkbenchV2
    base_family: MatrixFamily
    shifted_family: MatrixFamily
    train_matrices: tuple[Array, ...]
    train_targets: tuple[Array, ...]
    base_val_matrices: tuple[Array, ...]
    base_val_targets: tuple[Array, ...]
    shifted_eval_matrices: tuple[Array, ...]
    shifted_eval_targets: tuple[Array, ...]
    base_model: RidgeInverseApproximator
    showcase_model: RidgeInverseApproximator
    base_summary: TransferSummary
    zero_shot_summary: TransferSummary
    showcase_summary: TransferSummary
    base_records: tuple[TransferRecord, ...]
    zero_shot_records: tuple[TransferRecord, ...]
    showcase_records: tuple[TransferRecord, ...]
    adaptation_summaries: tuple[TransferSummary, ...]


def _orthogonal_basis(size: int, seed: int) -> Array:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((size, size)))
    return np.asarray(q, dtype=np.float64)


def _make_local_spd_family(
    *,
    name: str,
    basis: Array,
    rho_range: tuple[float, float],
    shift_range: tuple[float, float],
    wobble_range: tuple[float, float],
) -> MatrixFamily:
    grid = np.linspace(0.0, 1.0, basis.shape[0], dtype=np.float64)

    def generator(n: int, rng: np.random.Generator) -> Array:
        if int(n) != basis.shape[0]:
            raise ValueError(f"Family {name!r} only supports n={basis.shape[0]}")
        rho = rng.uniform(*rho_range)
        shift = rng.uniform(*shift_range)
        wobble = rng.uniform(*wobble_range)
        lam = shift + np.exp(3.0 * rho * grid) + wobble * np.sin(2.0 * np.pi * grid)
        lam = np.maximum(lam, 0.2)
        a = basis @ np.diag(lam) @ basis.T
        return np.asarray(0.5 * (a + a.T), dtype=np.float64)

    notes = (
        "Local SPD family with fixed eigenbasis; "
        f"rho in {rho_range}, shift in {shift_range}, wobble in {wobble_range}"
    )
    return MatrixFamily(name=name, generator=generator, notes=notes)


def build_transfer_workbench(
    config: TransferExperimentConfig,
) -> tuple[PreconditionerWorkbenchV2, MatrixFamily, MatrixFamily]:
    basis = _orthogonal_basis(config.matrix_size, config.basis_seed)
    base_family = _make_local_spd_family(
        name="Base Local SPD",
        basis=basis,
        rho_range=config.base_rho_range,
        shift_range=config.base_shift_range,
        wobble_range=config.base_wobble_range,
    )
    shifted_family = _make_local_spd_family(
        name="Shifted Local SPD",
        basis=basis,
        rho_range=config.shifted_rho_range,
        shift_range=config.shifted_shift_range,
        wobble_range=config.shifted_wobble_range,
    )
    workbench = PreconditionerWorkbenchV2(register_defaults=True)
    workbench.register_matrix_family(base_family)
    workbench.register_matrix_family(shifted_family)
    return workbench, base_family, shifted_family


def _build_targets(matrices: Sequence[Array], target: str) -> list[Array]:
    if target != "pinv":
        raise ValueError("This experiment is configured for target='pinv' only")
    return [np.linalg.pinv(a) for a in matrices]


def _gmres_result(
    a: Array,
    b: Array,
    preconditioner,
    *,
    tol: float,
    maxiter: int,
    restart: int,
) -> tuple[int, float]:
    operator = None if preconditioner is None else preconditioner.as_linear_operator()
    result = solve_gmres(a, b, operator, tol, maxiter, restart=restart)
    residual = float(np.linalg.norm(b - a @ result.x) / max(np.linalg.norm(b), 1e-12))
    return int(result.n_iter), residual


def evaluate_model_on_split(
    model: RidgeInverseApproximator,
    matrices: Sequence[Array],
    targets: Sequence[Array],
    *,
    regime: str,
    split: str,
    adaptation_samples: int,
    config: TransferExperimentConfig,
) -> tuple[tuple[TransferRecord, ...], TransferSummary]:
    rng = np.random.default_rng(config.rhs_seed)
    records: list[TransferRecord] = []
    for sample_index, (a, target) in enumerate(zip(matrices, targets)):
        approx = model.predict_inverse(a)
        ridge_preconditioner = model.as_preconditioner(a, name=regime)
        rhs = rng.standard_normal(a.shape[0])
        gmres_iterations, gmres_residual = _gmres_result(
            a,
            rhs,
            ridge_preconditioner,
            tol=config.gmres_tol,
            maxiter=config.gmres_maxiter,
            restart=config.gmres_restart,
        )
        no_preconditioner_iterations, _ = _gmres_result(
            a,
            rhs,
            None,
            tol=config.gmres_tol,
            maxiter=config.gmres_maxiter,
            restart=config.gmres_restart,
        )
        diagonal_iterations, _ = _gmres_result(
            a,
            rhs,
            diagonal_preconditioner(a),
            tol=config.gmres_tol,
            maxiter=config.gmres_maxiter,
            restart=config.gmres_restart,
        )
        denom = max(np.linalg.norm(target, ord="fro"), 1e-12)
        records.append(
            TransferRecord(
                regime=regime,
                split=split,
                sample_index=sample_index,
                matrix_kappa=float(condition_number(a, signed=False)),
                relative_fro_error=float(np.linalg.norm(approx - target, ord="fro") / denom),
                preconditioned_kappa=float(condition_number(approx @ a, signed=False)),
                gmres_iterations=gmres_iterations,
                gmres_residual=gmres_residual,
                no_preconditioner_iterations=no_preconditioner_iterations,
                diagonal_iterations=diagonal_iterations,
            )
        )

    matrix_kappa = np.asarray([record.matrix_kappa for record in records], dtype=np.float64)
    rel_error = np.asarray([record.relative_fro_error for record in records], dtype=np.float64)
    pre_kappa = np.asarray([record.preconditioned_kappa for record in records], dtype=np.float64)
    gmres_iterations = np.asarray([record.gmres_iterations for record in records], dtype=np.float64)
    gmres_residual = np.asarray([record.gmres_residual for record in records], dtype=np.float64)
    no_preconditioner = np.asarray(
        [record.no_preconditioner_iterations for record in records],
        dtype=np.float64,
    )
    diagonal = np.asarray([record.diagonal_iterations for record in records], dtype=np.float64)

    summary = TransferSummary(
        regime=regime,
        split=split,
        adaptation_samples=adaptation_samples,
        n_samples=len(records),
        matrix_kappa_mean=float(np.mean(matrix_kappa)),
        matrix_kappa_std=float(np.std(matrix_kappa)),
        relative_fro_error_mean=float(np.mean(rel_error)),
        relative_fro_error_std=float(np.std(rel_error)),
        preconditioned_kappa_mean=float(np.mean(pre_kappa)),
        preconditioned_kappa_std=float(np.std(pre_kappa)),
        gmres_iterations_mean=float(np.mean(gmres_iterations)),
        gmres_iterations_std=float(np.std(gmres_iterations)),
        gmres_residual_mean=float(np.mean(gmres_residual)),
        gmres_residual_std=float(np.std(gmres_residual)),
        no_preconditioner_iterations_mean=float(np.mean(no_preconditioner)),
        diagonal_iterations_mean=float(np.mean(diagonal)),
    )
    return tuple(records), summary


def _fit_model(
    matrices: Sequence[Array],
    targets: Sequence[Array],
    *,
    config: TransferExperimentConfig,
) -> RidgeInverseApproximator:
    return RidgeInverseApproximator(
        matrix_size=config.matrix_size,
        ridge=config.ridge_reg,
    ).fit(matrices, targets)


def run_transfer_experiment(
    config: TransferExperimentConfig | None = None,
) -> TransferBundle:
    if config is None:
        config = TransferExperimentConfig()

    apply_plot_style()
    workbench, base_family, shifted_family = build_transfer_workbench(config)

    train_matrices, train_targets = build_inverse_dataset(
        base_family,
        n=config.matrix_size,
        n_samples=config.base_train_samples,
        rng_seed=config.train_seed,
        target=config.target,
    )
    base_val_matrices, base_val_targets = build_inverse_dataset(
        base_family,
        n=config.matrix_size,
        n_samples=config.base_val_samples,
        rng_seed=config.val_seed,
        target=config.target,
    )
    shifted_eval_matrices, shifted_eval_targets = build_inverse_dataset(
        shifted_family,
        n=config.matrix_size,
        n_samples=config.shifted_eval_samples,
        rng_seed=config.shifted_seed,
        target=config.target,
    )

    base_model = _fit_model(train_matrices, train_targets, config=config)
    base_records, base_summary = evaluate_model_on_split(
        base_model,
        base_val_matrices,
        base_val_targets,
        regime="Base validation",
        split="base",
        adaptation_samples=0,
        config=config,
    )
    zero_shot_records, zero_shot_summary = evaluate_model_on_split(
        base_model,
        shifted_eval_matrices,
        shifted_eval_targets,
        regime="Shifted zero-shot",
        split="shifted",
        adaptation_samples=0,
        config=config,
    )

    adaptation_summaries: list[TransferSummary] = [zero_shot_summary]
    showcase_model = base_model
    showcase_records = zero_shot_records
    showcase_summary = zero_shot_summary

    for size in config.fine_tune_sizes:
        if size == 0:
            continue
        adapted_matrices = list(train_matrices) + list(shifted_eval_matrices[:size])
        adapted_targets = list(train_targets) + list(shifted_eval_targets[:size])
        adapted_model = _fit_model(adapted_matrices, adapted_targets, config=config)
        adapted_records, adapted_summary = evaluate_model_on_split(
            adapted_model,
            shifted_eval_matrices,
            shifted_eval_targets,
            regime=f"Shifted + fine-tune ({size})",
            split="shifted",
            adaptation_samples=size,
            config=config,
        )
        adaptation_summaries.append(adapted_summary)
        if size == config.showcase_fine_tune_size:
            showcase_model = adapted_model
            showcase_records = adapted_records
            showcase_summary = adapted_summary

    return TransferBundle(
        config=config,
        workbench=workbench,
        base_family=base_family,
        shifted_family=shifted_family,
        train_matrices=tuple(np.asarray(a) for a in train_matrices),
        train_targets=tuple(np.asarray(a) for a in train_targets),
        base_val_matrices=tuple(np.asarray(a) for a in base_val_matrices),
        base_val_targets=tuple(np.asarray(a) for a in base_val_targets),
        shifted_eval_matrices=tuple(np.asarray(a) for a in shifted_eval_matrices),
        shifted_eval_targets=tuple(np.asarray(a) for a in shifted_eval_targets),
        base_model=base_model,
        showcase_model=showcase_model,
        base_summary=base_summary,
        zero_shot_summary=zero_shot_summary,
        showcase_summary=showcase_summary,
        base_records=base_records,
        zero_shot_records=zero_shot_records,
        showcase_records=showcase_records,
        adaptation_summaries=tuple(
            sorted(adaptation_summaries, key=lambda item: item.adaptation_samples)
        ),
    )


def register_trained_model(
    workbench: PreconditionerWorkbenchV2,
    model: RidgeInverseApproximator,
    *,
    name: str,
) -> None:
    workbench.register_preconditioner(
        name,
        lambda a, trained_model=model, model_name=name: trained_model.as_preconditioner(
            a,
            name=model_name,
        ),
        max_n=model.matrix_size,
        notes="Ridge model learned in ridge_transfer experiment",
    )


def summaries_for_report(bundle: TransferBundle) -> tuple[TransferSummary, ...]:
    return (
        bundle.base_summary,
        bundle.zero_shot_summary,
        bundle.showcase_summary,
    )


def format_summary_table(summaries: Iterable[TransferSummary]) -> str:
    rows = list(summaries)
    header = (
        f"{'Regime':<24} {'Split':<8} {'Adapt':>5} {'Rel.Fro':>10} "
        f"{'cond(PA)':>10} {'GMRES':>8} {'None':>8} {'Diag':>8} {'Residual':>12}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row.regime:<24} {row.split:<8} {row.adaptation_samples:>5d} "
            f"{row.relative_fro_error_mean:>10.4f} "
            f"{row.preconditioned_kappa_mean:>10.4f} "
            f"{row.gmres_iterations_mean:>8.3f} "
            f"{row.no_preconditioner_iterations_mean:>8.3f} "
            f"{row.diagonal_iterations_mean:>8.3f} "
            f"{row.gmres_residual_mean:>12.3e}"
        )
    return "\n".join(lines)


def plot_regime_dashboard(
    summaries: Sequence[TransferSummary],
) -> tuple[plt.Figure, np.ndarray]:
    labels = [summary.regime for summary in summaries]
    x = np.arange(len(summaries), dtype=np.float64)
    colors = ["#2b6cb0", "#dd6b20", "#2f855a"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    ax = axes[0, 0]
    ax.bar(x, [summary.relative_fro_error_mean for summary in summaries], color=colors)
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.set_title("Relative Frobenius Error")
    ax.set_ylabel("mean")

    ax = axes[0, 1]
    ax.bar(x, [summary.preconditioned_kappa_mean for summary in summaries], color=colors)
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.set_title("cond(P @ A)")
    ax.set_ylabel("mean")

    ax = axes[1, 0]
    width = 0.22
    ax.bar(
        x - width,
        [summary.no_preconditioner_iterations_mean for summary in summaries],
        width=width,
        label="None",
        color="#718096",
    )
    ax.bar(
        x,
        [summary.diagonal_iterations_mean for summary in summaries],
        width=width,
        label="Diagonal",
        color="#a0aec0",
    )
    ax.bar(
        x + width,
        [summary.gmres_iterations_mean for summary in summaries],
        width=width,
        label="Ridge",
        color=colors,
    )
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.set_title("Mean GMRES Iterations")
    ax.set_ylabel("iterations")
    ax.legend()

    ax = axes[1, 1]
    ax.bar(x, [summary.gmres_residual_mean for summary in summaries], color=colors)
    ax.set_xticks(x, labels, rotation=12, ha="right")
    ax.set_title("Relative Residual")
    ax.set_ylabel("mean")
    ax.set_yscale("log")

    fig.tight_layout()
    return fig, axes


def plot_adaptation_curve(
    summaries: Sequence[TransferSummary],
) -> tuple[plt.Figure, np.ndarray]:
    ordered = sorted(summaries, key=lambda item: item.adaptation_samples)
    x = [summary.adaptation_samples for summary in ordered]
    no_preconditioner = ordered[0].no_preconditioner_iterations_mean
    diagonal = ordered[0].diagonal_iterations_mean

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].plot(x, [summary.relative_fro_error_mean for summary in ordered], marker="o")
    axes[0].set_title("Transfer Error vs Fine-Tune Size")
    axes[0].set_xlabel("shifted adaptation matrices")
    axes[0].set_ylabel("relative Fro error")

    axes[1].plot(x, [summary.preconditioned_kappa_mean for summary in ordered], marker="o")
    axes[1].set_title("cond(P @ A) vs Fine-Tune Size")
    axes[1].set_xlabel("shifted adaptation matrices")
    axes[1].set_ylabel("mean cond(P @ A)")

    axes[2].plot(x, [summary.gmres_iterations_mean for summary in ordered], marker="o")
    axes[2].axhline(no_preconditioner, color="#718096", linestyle="--", label="None")
    axes[2].axhline(diagonal, color="#a0aec0", linestyle=":", label="Diagonal")
    axes[2].set_title("GMRES Iterations vs Fine-Tune Size")
    axes[2].set_xlabel("shifted adaptation matrices")
    axes[2].set_ylabel("mean iterations")
    axes[2].legend()

    fig.tight_layout()
    return fig, axes


def plot_per_sample_transfer(
    zero_shot_records: Sequence[TransferRecord],
    tuned_records: Sequence[TransferRecord],
) -> tuple[plt.Figure, np.ndarray]:
    zero_map = {record.sample_index: record for record in zero_shot_records}
    tuned_map = {record.sample_index: record for record in tuned_records}
    common_indices = sorted(zero_map)
    order = sorted(
        common_indices,
        key=lambda idx: zero_map[idx].relative_fro_error,
        reverse=True,
    )

    x = np.arange(len(order), dtype=np.float64)
    zero_errors = [zero_map[idx].relative_fro_error for idx in order]
    tuned_errors = [tuned_map[idx].relative_fro_error for idx in order]
    zero_iters = [zero_map[idx].gmres_iterations for idx in order]
    tuned_iters = [tuned_map[idx].gmres_iterations for idx in order]
    diagonal_iters = [zero_map[idx].diagonal_iterations for idx in order]
    no_pre_iters = [zero_map[idx].no_preconditioner_iterations for idx in order]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(x, zero_errors, marker="o", label="Zero-shot", color="#dd6b20")
    axes[0].plot(
        x,
        tuned_errors,
        marker="o",
        label="Fine-tune",
        color="#2f855a",
    )
    axes[0].set_title("Per-Sample Relative Error on Shifted Split")
    axes[0].set_xlabel("samples sorted by zero-shot difficulty")
    axes[0].set_ylabel("relative Fro error")
    axes[0].legend()

    axes[1].plot(x, no_pre_iters, label="None", color="#718096", linestyle="--")
    axes[1].plot(x, diagonal_iters, label="Diagonal", color="#a0aec0", linestyle=":")
    axes[1].plot(x, zero_iters, label="Zero-shot", color="#dd6b20")
    axes[1].plot(x, tuned_iters, label="Fine-tune", color="#2f855a")
    axes[1].set_title("Per-Sample GMRES Iterations on Shifted Split")
    axes[1].set_xlabel("samples sorted by zero-shot difficulty")
    axes[1].set_ylabel("iterations")
    axes[1].legend()

    fig.tight_layout()
    return fig, axes


def describe_registered_families(bundle: TransferBundle) -> str:
    relevant = {
        bundle.base_family.name,
        bundle.shifted_family.name,
    }
    lines = []
    for family in bundle.workbench.list_matrix_families():
        if family.name in relevant:
            lines.append(f"- {family.name}: {family.notes}")
    return "\n".join(lines)


def main() -> None:
    bundle = run_transfer_experiment()
    print(format_summary_table(summaries_for_report(bundle)))
    print()
    print("Fine-tune curve:")
    print(format_summary_table(bundle.adaptation_summaries))


if __name__ == "__main__":
    main()
