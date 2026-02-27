from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import lu_factor, lu_solve, circulant
from scipy.sparse.linalg import LinearOperator


Array = NDArray[np.float64]
Vector = NDArray[np.float64]


class Preconditioner:
    def __init__(
        self,
        name: str,
        apply_vec: Callable[[Vector], Vector],
        size: int,
        matrix: Optional[Array] = None,
    ) -> None:
        self.name = name
        self._apply_vec = apply_vec
        self.size = size
        self._matrix = matrix

    def apply(self, x: Vector) -> Vector:
        return self._apply_vec(x)

    def apply_to_matrix(self, A: Array) -> Array:
        if self._matrix is not None:
            return self._matrix @ A
        cols = [self.apply(A[:, j]) for j in range(A.shape[1])]
        return np.column_stack(cols)

    def as_linear_operator(self) -> LinearOperator:
        n = self.size
        return LinearOperator((n, n), matvec=self.apply, dtype=np.float64)


@dataclass(frozen=True)
class PreconditionerFactory:
    name: str
    builder: Callable[[Array], Preconditioner]
    max_n: Optional[int] = None
    notes: str = ""

    def build(self, A: Array) -> Preconditioner:
        return self.builder(A)

    def supports(self, n: int) -> bool:
        return self.max_n is None or n <= self.max_n


def identity_preconditioner(A: Array) -> Preconditioner:
    n = A.shape[0]
    return Preconditioner("None", lambda x: x.copy(), n)


def diagonal_preconditioner(A: Array, rcond: float = 1e-12) -> Preconditioner:
    n = A.shape[0]
    diag = np.diag(A)
    scale = np.where(np.abs(diag) > rcond, 1.0 / diag, 1.0)

    def apply(x: Vector) -> Vector:
        return scale * x

    return Preconditioner("Diagonal", apply, n)


def lu_preconditioner(A: Array) -> Preconditioner:
    n = A.shape[0]
    lu, piv = lu_factor(A)

    def apply(x: Vector) -> Vector:
        return lu_solve((lu, piv), x)

    return Preconditioner("LU", apply, n)


def _circulant_average_diagonals(A: Array) -> Vector:
    n = A.shape[0]
    c = np.zeros(n, dtype=np.float64)
    for k in range(n):
        idx = (np.arange(n) + k) % n
        c[k] = np.mean(A[np.arange(n), idx])
    return c


def circulant_preconditioner(A: Array, rcond: float = 1e-12) -> Preconditioner:
    n = A.shape[0]
    c = _circulant_average_diagonals(A)
    eigvals = np.fft.fft(c)
    max_abs = np.max(np.abs(eigvals))
    safe = np.abs(eigvals) > (rcond * max_abs)
    inv_eigs = np.zeros_like(eigvals, dtype=np.complex128)
    inv_eigs[safe] = 1.0 / eigvals[safe]

    def apply(x: Vector) -> Vector:
        fx = np.fft.fft(x)
        y = np.fft.ifft(inv_eigs * fx)
        return np.asarray(np.real(y), dtype=np.float64)

    P = None
    if n <= 256:
        C = circulant(c)
        try:
            P = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            P = None
    return Preconditioner("Circulant", apply, n, matrix=P)


def svd_preconditioner(A: Array, rcond: float = 1e-12) -> Preconditioner:
    n = A.shape[0]
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    cutoff = rcond * np.max(s)
    inv_s = np.where(s > cutoff, 1.0 / s, 0.0)

    def apply(x: Vector) -> Vector:
        y = U.T @ x
        y = inv_s * y
        return Vt.T @ y

    return Preconditioner("SVD", apply, n)


def default_preconditioners() -> Iterable[PreconditionerFactory]:
    return [
        PreconditionerFactory("None", identity_preconditioner, notes="Baseline"),
        PreconditionerFactory("Diagonal", diagonal_preconditioner, notes="Jacobi"),
        PreconditionerFactory("LU", lu_preconditioner, max_n=512, notes="Dense LU; limited size"),
        PreconditionerFactory("Circulant", circulant_preconditioner, notes="FFT-based"),
        PreconditionerFactory("SVD", svd_preconditioner, max_n=256, notes="Expensive; limited size"),
    ]
