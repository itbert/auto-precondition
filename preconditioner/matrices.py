from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import hilbert, toeplitz


Array = NDArray[np.float64]
T = TypeVar("T")


@dataclass(frozen=True)
class MatrixFamily:
    name: str
    generator: Callable[[int, np.random.Generator], Array]
    notes: str = ""


def random_dense(n: int, rng: np.random.Generator) -> Array:
    return rng.standard_normal((n, n))


def diagonally_dominant(n: int,
                        rng: np.random.Generator,
                        margin: float = 1.0) -> Array:
    A = rng.standard_normal((n, n))
    row_sums = np.sum(np.abs(A), axis=1)
    A[np.arange(n), np.arange(n)] = row_sums + margin
    return A


def spd_matrix(n: int, rng: np.random.Generator, shift: float = 1e-2) -> Array:
    B = rng.standard_normal((n, n))
    return B.T @ B + shift * np.eye(n)


def hilbert_matrix(n: int, _rng: np.random.Generator) -> Array:
    return hilbert(n)


def toeplitz_matrix(n: int, rng: np.random.Generator) -> Array:
    c = rng.standard_normal(n)
    return toeplitz(c)


def random_spectrum(
    n: int,
    rng: np.random.Generator,
    kappa: float = 1e6,
) -> Array:
    abs_kappa = float(abs(kappa))
    if abs_kappa < 1.0:
        abs_kappa = 1.0 / max(abs_kappa, 1e-12)
    eigvals = np.geomspace(abs_kappa, 1.0, num=n)
    if kappa < 0:
        eigvals[-1] *= -1.0
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    A = Q @ np.diag(eigvals) @ Q.T
    return 0.5 * (A + A.T)


def uniform_size_grid(
    min_size: int = 16,
    max_size: int = 512,
    count: int = 8,
) -> Tuple[int, ...]:
    if count <= 0:
        raise ValueError("count must be positive")
    if min_size <= 0 or max_size < min_size:
        raise ValueError("invalid [min_size, max_size] range")
    if count == 1:
        return (int(min_size),)
    values = np.linspace(min_size, max_size, num=count)
    sizes = sorted({int(round(v)) for v in values})
    if sizes[0] != min_size:
        sizes[0] = min_size
    if sizes[-1] != max_size:
        sizes[-1] = max_size
    return tuple(sizes)


def uniform_ordered_selection(items: Sequence[T], count: int) -> List[T]:
    if count <= 0:
        raise ValueError("count must be positive")
    if count >= len(items):
        return list(items)
    indices = np.linspace(0, len(items) - 1, num=count, dtype=int)
    selected: List[T] = []
    used: set[int] = set()
    for idx in indices:
        i = int(idx)
        if i not in used:
            selected.append(items[i])
            used.add(i)
    if len(selected) < count:
        for i, item in enumerate(items):
            if i not in used:
                selected.append(item)
                used.add(i)
            if len(selected) == count:
                break
    return selected[:count]


def default_matrix_families() -> List[MatrixFamily]:
    return [
        MatrixFamily("Random Dense", random_dense, "Gaussian entries"),
        MatrixFamily("Diag Dominant", diagonally_dominant, "Strict diagonal dominance"),
        MatrixFamily("SPD", spd_matrix, "B^T B + shift * I"),
        MatrixFamily("Hilbert", hilbert_matrix, "Ill-conditioned"),
        MatrixFamily("Toeplitz", toeplitz_matrix, "Structured Toeplitz"),
        MatrixFamily(
            "Random Spectrum",
            lambda n, rng: random_spectrum(n, rng, 1e4),
            "Controllable signed spectral ratio",
        ),
    ]


def generate_matrices(
    family: MatrixFamily,
    sizes: Iterable[int],
    n_samples: int,
    rng: np.random.Generator,
) -> Iterable[Tuple[int, Array]]:
    for n in sizes:
        for _ in range(n_samples):
            yield n, family.generator(n, rng)
