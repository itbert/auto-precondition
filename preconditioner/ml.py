from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

try:
    from .matrices import MatrixFamily
    from .metrics import condition_number
    from .preconditioners import Preconditioner
except ImportError:
    from matrices import MatrixFamily  # type: ignore
    from metrics import condition_number  # type: ignore
    from preconditioners import Preconditioner  # type: ignore


Array = NDArray[np.generic]
TargetKind = Literal["inverse", "pinv", "diagonal_inverse"]


@dataclass(frozen=True)
class InverseApproximationMetrics:
    relative_fro_error_mean: float
    relative_fro_error_std: float
    preconditioned_kappa_mean: float
    preconditioned_kappa_std: float


class RidgeInverseApproximator:
    def __init__(
        self,
        *,
        matrix_size: int,
        ridge: float = 1e-4,
        add_bias: bool = True,
    ) -> None:
        if matrix_size <= 0:
            raise ValueError("matrix_size must be positive")
        self.matrix_size = int(matrix_size)
        self.ridge = float(ridge)
        self.add_bias = bool(add_bias)
        self._weights: Optional[Array] = None
        self._dtype: np.dtype = np.dtype(np.float64)

    def fit(self, matrices: Sequence[Array], targets: Sequence[Array]) -> "RidgeInverseApproximator":
        if len(matrices) == 0 or len(targets) == 0:
            raise ValueError("training data must be non-empty")
        if len(matrices) != len(targets):
            raise ValueError("matrices and targets must have equal length")

        dtype = np.result_type(*(np.asarray(A).dtype for A in matrices), *(np.asarray(T).dtype for T in targets))
        self._dtype = np.dtype(dtype)
        X = np.column_stack([self._features(np.asarray(A, dtype=self._dtype)) for A in matrices])
        Y = np.column_stack([np.asarray(T, dtype=self._dtype).reshape(-1) for T in targets])
        gram = X @ X.conj().T
        reg = self.ridge * np.eye(gram.shape[0], dtype=self._dtype)
        rhs = Y @ X.conj().T
        self._weights = np.linalg.solve(gram + reg, rhs.T).T
        return self

    def predict_inverse(self, A: Array) -> Array:
        self._check_fitted()
        x = self._features(A)
        y = self._weights @ x
        return y.reshape(self.matrix_size, self.matrix_size)

    def as_preconditioner(self, A: Array, *, name: str = "RidgeApprox") -> Preconditioner:
        P = self.predict_inverse(A)

        def apply(x: NDArray[np.generic]) -> NDArray[np.generic]:
            return P @ x

        return Preconditioner(
            name=name,
            apply_vec=apply,
            size=self.matrix_size,
            matrix=P,
            dtype=P.dtype,
        )

    def _features(self, A: Array) -> NDArray[np.generic]:
        if A.shape != (self.matrix_size, self.matrix_size):
            raise ValueError(
                f"Expected shape {(self.matrix_size, self.matrix_size)}, got {A.shape}"
            )
        flat = A.reshape(-1).astype(self._dtype, copy=False)
        if not self.add_bias:
            return flat
        return np.concatenate([flat, np.array([1.0], dtype=self._dtype)])

    def _check_fitted(self) -> None:
        if self._weights is None:
            raise RuntimeError("model is not fitted")


def build_inverse_dataset(
    family: MatrixFamily,
    *,
    n: int,
    n_samples: int,
    rng_seed: int = 0,
    target: TargetKind = "pinv",
) -> tuple[list[Array], list[Array]]:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    rng = np.random.default_rng(rng_seed)
    matrices: list[Array] = []
    targets: list[Array] = []

    for _ in range(n_samples):
        A = family.generator(n, rng)
        if target == "inverse":
            T = np.linalg.inv(A)
        elif target == "pinv":
            T = np.linalg.pinv(A)
        elif target == "diagonal_inverse":
            d = np.diag(A)
            inv_d = np.where(np.abs(d) > 1e-12, 1.0 / d, 1.0)
            T = np.diag(inv_d)
        else:
            raise ValueError(f"Unknown target kind: {target!r}")
        matrices.append(np.asarray(A))
        targets.append(np.asarray(T))
    return matrices, targets


def evaluate_inverse_model(
    model: RidgeInverseApproximator,
    matrices: Iterable[Array],
    targets: Iterable[Array],
    *,
    signed_kappa: bool = True,
) -> InverseApproximationMetrics:
    rel_errors: list[float] = []
    kappas: list[float] = []
    for A, T in zip(matrices, targets):
        P = model.predict_inverse(A)
        denom = np.linalg.norm(T, ord="fro")
        rel = np.linalg.norm(P - T, ord="fro") / max(denom, 1e-12)
        rel_errors.append(float(rel))
        kappas.append(
            float(
                condition_number(
                    P @ A,
                    signed=signed_kappa and not np.iscomplexobj(P @ A),
                )
            )
        )
    return InverseApproximationMetrics(
        relative_fro_error_mean=float(np.mean(rel_errors)) if rel_errors else float("nan"),
        relative_fro_error_std=float(np.std(rel_errors)) if rel_errors else float("nan"),
        preconditioned_kappa_mean=float(np.mean(kappas)) if kappas else float("nan"),
        preconditioned_kappa_std=float(np.std(kappas)) if kappas else float("nan"),
    )
