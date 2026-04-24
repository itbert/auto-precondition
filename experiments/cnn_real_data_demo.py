from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import tempfile

import numpy as np

MPLCONFIGDIR = Path(tempfile.gettempdir()) / "auto_preconditioner_mpl"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator


ROOT_DIR = Path(__file__).resolve().parents[1]
PRECONDITIONER_SRC = ROOT_DIR / "preconditioner"
TESTS_DIR = ROOT_DIR / "tests"
for path in (PRECONDITIONER_SRC, TESTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_real_data import (  # noqa: E402
    DATASET_ALIASES,
    build_impedance_matrix,
    build_rhs,
    load_compressed_dataset,
    resolve_data_dir,
)
from cnn import ConvolutionalInverseApproximator  # noqa: E402
from solvers import solve_gmres  # noqa: E402


@dataclass(frozen=True)
class DemoRecord:
    frequency_hz: float
    gmres_none_iter: int
    gmres_diag_iter: int
    gmres_cnn_iter: int
    gmres_none_residual: float
    gmres_diag_residual: float
    gmres_cnn_residual: float
    diag_consistency: float
    cnn_consistency: float
    cnn_relative_inverse_error: float


def jacobi_inverse(A: np.ndarray) -> np.ndarray:
    diag = np.diag(A)
    inv_diag = np.where(np.abs(diag) > 1e-12, 1.0 / diag, 1.0)
    P = np.zeros_like(A, dtype=np.complex128)
    idx = np.arange(A.shape[0])
    P[idx, idx] = inv_diag
    return P


def relative_frobenius_error(predicted: np.ndarray, target: np.ndarray) -> float:
    return float(
        np.linalg.norm(predicted - target, ord="fro")
        / max(np.linalg.norm(target, ord="fro"), 1e-12)
    )


def consistency_error(P: np.ndarray, A: np.ndarray) -> float:
    eye = np.eye(A.shape[0], dtype=A.dtype)
    return float(np.linalg.norm(P @ A - eye, ord="fro") / A.shape[0])


def uniform_train_test_split(freqs: np.ndarray, train_count: int, test_count: int) -> tuple[np.ndarray, np.ndarray]:
    if train_count <= 0 or test_count <= 0:
        raise ValueError("train_count and test_count must be positive")
    if len(freqs) < train_count + test_count:
        raise ValueError("not enough frequencies for the requested split")
    train_idx = np.linspace(0, len(freqs) - 1, num=train_count, dtype=int)
    mask = np.zeros(len(freqs), dtype=bool)
    mask[np.unique(train_idx)] = True
    train_freqs = freqs[mask]
    test_freqs = freqs[~mask][:test_count]
    if len(test_freqs) < test_count:
        raise ValueError("unable to build the requested test split")
    return train_freqs, test_freqs


def plot_training(history, output_dir: Path) -> None:
    epochs = np.arange(1, len(history.train_loss) + 1)
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, history.train_loss, label="train total", linewidth=2)
    plt.plot(epochs, history.val_loss, label="val total", linewidth=2)
    plt.plot(epochs, history.train_target_loss, label="train target", linestyle="--")
    plt.plot(epochs, history.train_consistency_loss, label="train consistency", linestyle=":")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("CNNApprox training history")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=180)
    plt.close()


def plot_iterations(records: list[DemoRecord], output_dir: Path) -> None:
    freqs_mhz = np.asarray([record.frequency_hz for record in records], dtype=np.float64) / 1e6
    none_iter = np.asarray([record.gmres_none_iter for record in records], dtype=np.float64)
    diag_iter = np.asarray([record.gmres_diag_iter for record in records], dtype=np.float64)
    cnn_iter = np.asarray([record.gmres_cnn_iter for record in records], dtype=np.float64)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs_mhz, none_iter, marker="o", label="None", linewidth=1.8)
    plt.plot(freqs_mhz, diag_iter, marker="s", label="Diagonal", linewidth=1.8)
    plt.plot(freqs_mhz, cnn_iter, marker="^", label="CNNApprox", linewidth=2.2)
    plt.xlabel("Frequency, MHz")
    plt.ylabel("GMRES iterations")
    plt.title("GMRES iterations on held-out frequencies")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "gmres_iterations_vs_frequency.png", dpi=180)
    plt.close()


def plot_consistency(records: list[DemoRecord], output_dir: Path) -> None:
    freqs_mhz = np.asarray([record.frequency_hz for record in records], dtype=np.float64) / 1e6
    diag_cons = np.asarray([record.diag_consistency for record in records], dtype=np.float64)
    cnn_cons = np.asarray([record.cnn_consistency for record in records], dtype=np.float64)

    plt.figure(figsize=(10, 4))
    plt.semilogy(freqs_mhz, diag_cons, marker="s", label="Diagonal", linewidth=1.8)
    plt.semilogy(freqs_mhz, cnn_cons, marker="^", label="CNNApprox", linewidth=2.2)
    plt.xlabel("Frequency, MHz")
    plt.ylabel(r"$||PA - I||_F / n$")
    plt.title("Preconditioner consistency")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "preconditioner_consistency_vs_frequency.png", dpi=180)
    plt.close()


def plot_inverse_error(records: list[DemoRecord], output_dir: Path) -> None:
    freqs_mhz = np.asarray([record.frequency_hz for record in records], dtype=np.float64) / 1e6
    errors = np.asarray([record.cnn_relative_inverse_error for record in records], dtype=np.float64)

    plt.figure(figsize=(10, 4))
    plt.semilogy(freqs_mhz, errors, marker="^", linewidth=2.2)
    plt.xlabel("Frequency, MHz")
    plt.ylabel("Relative Frobenius error")
    plt.title("CNN inverse approximation error")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "inverse_relative_error_vs_frequency.png", dpi=180)
    plt.close()


def write_csv(path: Path, rows: list[DemoRecord]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def run_demo(
    *,
    dataset: str = "large",
    data_dir: Path | None = None,
    res_min: float = 38e6,
    res_max: float = 42e6,
    freq_limit: int = 24,
    train_count: int = 16,
    test_count: int = 8,
    hidden_channels: int = 12,
    basis_rank: int = 6,
    epochs: int = 10,
    batch_size: int = 2,
    learning_rate: float = 2e-3,
    consistency_weight: float = 0.3,
    output_dir: Path | None = None,
) -> tuple[list[DemoRecord], Path]:
    data_path = resolve_data_dir(dataset=dataset, data_dir=data_dir)
    dataset_obj = load_compressed_dataset(data_path)
    freqs = dataset_obj.selected_freqs(res_min=res_min, res_max=res_max, limit=freq_limit)
    train_freqs, test_freqs = uniform_train_test_split(freqs, train_count, test_count)

    train_matrices = [build_impedance_matrix(dataset_obj, float(freq)) for freq in train_freqs]
    train_targets = [np.linalg.inv(A) for A in train_matrices]
    model = ConvolutionalInverseApproximator(
        matrix_size=dataset_obj.n,
        hidden_channels=hidden_channels,
        basis_rank=basis_rank,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        consistency_weight=consistency_weight,
    ).fit(train_matrices, train_targets, verbose=True)

    records: list[DemoRecord] = []
    for freq in test_freqs:
        A = build_impedance_matrix(dataset_obj, float(freq))
        rhs = build_rhs(dataset_obj, float(freq))
        target = np.linalg.inv(A)
        P_diag = jacobi_inverse(A)
        P_cnn = model.predict_inverse(A)

        none_result = solve_gmres(A, rhs, None, 1e-6, 200, restart=40)
        diag_result = solve_gmres(
            A,
            rhs,
            LinearOperator(A.shape, matvec=lambda x, P=P_diag: P @ x, dtype=A.dtype),
            1e-6,
            200,
            restart=40,
        )
        cnn_result = solve_gmres(
            A,
            rhs,
            LinearOperator(A.shape, matvec=lambda x, P=P_cnn: P @ x, dtype=A.dtype),
            1e-6,
            200,
            restart=40,
        )

        records.append(
            DemoRecord(
                frequency_hz=float(freq),
                gmres_none_iter=none_result.n_iter,
                gmres_diag_iter=diag_result.n_iter,
                gmres_cnn_iter=cnn_result.n_iter,
                gmres_none_residual=float(np.linalg.norm(rhs - A @ none_result.x) / np.linalg.norm(rhs)),
                gmres_diag_residual=float(np.linalg.norm(rhs - A @ diag_result.x) / np.linalg.norm(rhs)),
                gmres_cnn_residual=float(np.linalg.norm(rhs - A @ cnn_result.x) / np.linalg.norm(rhs)),
                diag_consistency=consistency_error(P_diag, A),
                cnn_consistency=consistency_error(P_cnn, A),
                cnn_relative_inverse_error=relative_frobenius_error(P_cnn, target),
            )
        )

    if output_dir is None:
        output_dir = ROOT_DIR / "experiments" / "cnn_real_data_demo_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(output_dir / "records.csv", records)
    plot_training(model.history, output_dir)
    plot_iterations(records, output_dir)
    plot_consistency(records, output_dir)
    plot_inverse_error(records, output_dir)

    summary = {
        "dataset": dataset_obj.data_dir.as_posix(),
        "matrix_size": dataset_obj.n,
        "train_frequencies_hz": [float(freq) for freq in train_freqs],
        "test_frequencies_hz": [float(freq) for freq in test_freqs],
        "architecture": model.describe_architecture(),
        "gmres_none_mean_iter": float(np.mean([record.gmres_none_iter for record in records])),
        "gmres_diag_mean_iter": float(np.mean([record.gmres_diag_iter for record in records])),
        "gmres_cnn_mean_iter": float(np.mean([record.gmres_cnn_iter for record in records])),
        "diag_consistency_mean": float(np.mean([record.diag_consistency for record in records])),
        "cnn_consistency_mean": float(np.mean([record.cnn_consistency for record in records])),
        "cnn_relative_inverse_error_mean": float(
            np.mean([record.cnn_relative_inverse_error for record in records])
        ),
    }
    with (output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    return records, output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate CNNApprox on real impedance data.")
    parser.add_argument("--dataset", choices=sorted(DATASET_ALIASES), default="large")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--res-min", type=float, default=38e6)
    parser.add_argument("--res-max", type=float, default=42e6)
    parser.add_argument("--freq-limit", type=int, default=24)
    parser.add_argument("--train-count", type=int, default=16)
    parser.add_argument("--test-count", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=12)
    parser.add_argument("--basis-rank", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--consistency-weight", type=float, default=0.3)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    records, output_dir = run_demo(
        dataset=args.dataset,
        data_dir=args.data_dir,
        res_min=args.res_min,
        res_max=args.res_max,
        freq_limit=args.freq_limit,
        train_count=args.train_count,
        test_count=args.test_count,
        hidden_channels=args.hidden_channels,
        basis_rank=args.basis_rank,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        consistency_weight=args.consistency_weight,
        output_dir=args.output_dir,
    )
    print(f"Saved {len(records)} evaluation rows to {output_dir}")
