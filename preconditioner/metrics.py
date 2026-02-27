from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


def signed_symmetric_condition_number(
    A: Array,
    *,
    zero_tol: float = 1e-12,
) -> float:
    A_sym = 0.5 * (A + A.T)
    try:
        eigvals = np.linalg.eigvalsh(A_sym)
    except np.linalg.LinAlgError:
        return float(np.inf)
    lam_max = float(eigvals[-1])
    lam_min = float(eigvals[0])
    if abs(lam_min) <= zero_tol:
        sign = 1.0 if lam_max >= 0 else -1.0
        return float(np.copysign(np.inf, sign))
    return lam_max / lam_min


def norm2_condition_number(A: Array) -> float:
    try:
        return float(np.linalg.cond(A))
    except np.linalg.LinAlgError:
        return float(np.inf)


def condition_number(
    A: Array,
    *,
    signed: bool = True,
    zero_tol: float = 1e-12,
) -> float:
    if signed:
        return signed_symmetric_condition_number(A, zero_tol=zero_tol)
    return norm2_condition_number(A)


def delta_kappa(kappa_pre: float, kappa_base: float) -> float:
    return kappa_pre - kappa_base


def ratio_kappa(
    kappa_pre: float,
    kappa_base: float,
    *,
    zero_tol: float = 1e-12,
) -> float:
    if abs(kappa_base) <= zero_tol:
        sign = 1.0 if kappa_pre >= 0 else -1.0
        return float(np.copysign(np.inf, sign))
    return kappa_pre / kappa_base


@dataclass(frozen=True)
class KappaMetrics:
    kappa: float
    kappa_pre: float
    delta: float
    ratio: float


def compute_kappa_metrics(kappa_base: float, kappa_pre: float) -> KappaMetrics:
    return KappaMetrics(
        kappa=kappa_base,
        kappa_pre=kappa_pre,
        delta=delta_kappa(kappa_pre, kappa_base),
        ratio=ratio_kappa(kappa_pre, kappa_base),
    )
