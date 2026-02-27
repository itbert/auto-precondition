from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
import inspect
from scipy.sparse.linalg import gmres, lgmres, bicgstab, LinearOperator


Array = NDArray[np.float64]
Vector = NDArray[np.float64]


@dataclass(frozen=True)
class SolverResult:
    x: Vector
    info: int
    n_iter: int
    residual_norm: float
    residuals: Tuple[float, ...]

    @property
    def converged(self) -> bool:
        return self.info == 0


@dataclass(frozen=True)
class SolverSpec:
    name: str
    solve: Callable[[Array,
                     Vector,
                     Optional[LinearOperator],
                     float,
                     int], SolverResult]


def _callback_builder(
    A: Array,
    b: Vector,
    residuals: list[float],
) -> Callable[[Vector | float], None]:
    def callback(arg: Vector | float) -> None:
        if np.ndim(arg) == 0:
            residuals.append(float(arg))
        else:
            r = b - A @ np.asarray(arg)
            residuals.append(float(np.linalg.norm(r)))

    return callback


def _supports_param(func, name: str) -> bool:
    return name in inspect.signature(func).parameters


def _build_tol_kwargs(func, tol: float) -> dict:
    if _supports_param(func, "rtol"):
        return {"rtol": tol, "atol": 0.0}
    return {"tol": tol}


def solve_gmres(
    A: Array,
    b: Vector,
    M: Optional[LinearOperator],
    tol: float,
    maxiter: int,
    restart: Optional[int] = None,
) -> SolverResult:
    residuals: list[float] = []
    cb = _callback_builder(A, b, residuals)
    kwargs = _build_tol_kwargs(gmres, tol)
    if _supports_param(gmres, "restart") and restart is not None:
        kwargs["restart"] = restart
    if _supports_param(gmres, "maxiter"):
        kwargs["maxiter"] = maxiter
    if _supports_param(gmres, "M") and M is not None:
        kwargs["M"] = M
    if _supports_param(gmres, "callback"):
        kwargs["callback"] = cb
    if _supports_param(gmres, "callback_type"):
        kwargs["callback_type"] = "pr_norm"

    x, info = gmres(A, b, **kwargs)
    final_res = float(np.linalg.norm(b - A @ x))
    return SolverResult(x, info, len(residuals), final_res, tuple(residuals))


def solve_lgmres(
    A: Array,
    b: Vector,
    M: Optional[LinearOperator],
    tol: float,
    maxiter: int,
) -> SolverResult:
    residuals: list[float] = []
    cb = _callback_builder(A, b, residuals)
    kwargs = _build_tol_kwargs(lgmres, tol)
    if _supports_param(lgmres, "maxiter"):
        kwargs["maxiter"] = maxiter
    if _supports_param(lgmres, "M") and M is not None:
        kwargs["M"] = M
    if _supports_param(lgmres, "callback"):
        kwargs["callback"] = cb

    x, info = lgmres(A, b, **kwargs)
    final_res = float(np.linalg.norm(b - A @ x))
    return SolverResult(x, info, len(residuals), final_res, tuple(residuals))


def solve_bicgstab(
    A: Array,
    b: Vector,
    M: Optional[LinearOperator],
    tol: float,
    maxiter: int,
) -> SolverResult:
    residuals: list[float] = []
    cb = _callback_builder(A, b, residuals)
    kwargs = _build_tol_kwargs(bicgstab, tol)
    if _supports_param(bicgstab, "maxiter"):
        kwargs["maxiter"] = maxiter
    if _supports_param(bicgstab, "M") and M is not None:
        kwargs["M"] = M
    if _supports_param(bicgstab, "callback"):
        kwargs["callback"] = cb

    x, info = bicgstab(A, b, **kwargs)
    final_res = float(np.linalg.norm(b - A @ x))
    return SolverResult(x, info, len(residuals), final_res, tuple(residuals))


def default_solvers(
    restart: Optional[int] = None,
) -> Tuple[SolverSpec, ...]:
    def gmres_wrapper(A, b, M, tol, maxiter):
        return solve_gmres(A, b, M, tol, maxiter, restart=restart)

    return (
        SolverSpec("GMRES", gmres_wrapper),
        SolverSpec("LGMRES", solve_lgmres),
        SolverSpec("BiCGSTAB", solve_bicgstab),
    )
