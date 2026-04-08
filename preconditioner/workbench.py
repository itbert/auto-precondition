""" API для генерации матриц, построения предобуславливателей,
    запуска серийных экспериментов и оценки метрик
    (kappa, delta, ratio), итераций и времени: как в онбординге, но не хендмейд

1. PreconditionerWorkbench
   Готовая конфигурация с дефолтными гиперпараметрами
2. PreconditionerWorkbenchV2
   Абстрактный конструктор с реестрами: можно подключать свои матрицы,
   предобуславливатели и солверы по имени или объектом. Типо 'interface', который можно override-ить

Пример:
    >>> import numpy as np
    >>> from preconditioner import Preconditioner, PreconditionerWorkbenchV2
    >>>
    >>> wb = PreconditionerWorkbenchV2()
    >>>
    >>> wb.register_matrix_generator(
    ...     "SupaPupaMatrixFamily",
    ...     lambda n, rng: rng.standard_normal((n, n)) + 2.0 * np.eye(n),
    ...     notes="Dense + shift, эксп. №1",
    ... )
    >>>
    >>> def my_builder(A):
    ...     n = A.shape[0]
    ...     return Preconditioner("GigaDiagonal", lambda x: x / np.diag(A), n)
    ...
    >>> wb.register_preconditioner("GigaDiagonal", my_builder)
    >>>
    >>> A = wb.make_matrix("SupaPupaMatrixFamily", n=64)
    >>> res = wb.evaluate_matrix(A, preconditioner="GigaDiagonal", solver="GMRES")
    >>> res.kappa, res.delta_kappa
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Iterable, Optional, Protocol, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .experiments import (
    AggregatedRecord,
    ExperimentConfig,
    ExperimentRecord,
    KappaTimingRecord,
    aggregate_records,
    aggregate_timing_by_kappa,
    run_experiments,
)
from .matrices import (
    MatrixFamily,
    default_matrix_families,
    random_spectrum,
    uniform_ordered_selection,
    uniform_size_grid,
)
from .metrics import compute_kappa_metrics, condition_number
from .plotting import (
    plot_delta_kappa_vs_kappa,
    plot_delta_kappa_vs_n,
    plot_iterations_vs_n,
    plot_time_vs_kappa,
    plot_time_vs_n,
)
from .preconditioners import Preconditioner, PreconditionerFactory, default_preconditioners
from .solvers import SolverResult, SolverSpec, default_solvers


Array = NDArray[np.float64]
Vector = NDArray[np.float64]
MatrixFamilyLike = Union[MatrixFamily, str]
PreconditionerLike = Union[PreconditionerFactory, str]
SolverLike = Union[SolverSpec, str]


class SolverFn(Protocol):
    def __call__(
        self,
        A: Array,
        b: Vector,
        M,
        tol: float,
        maxiter: int,
    ) -> SolverResult: ...


class PreconditionerBuilder(Protocol):
    def __call__(self, A: Array) -> Preconditioner: ...


@dataclass(frozen=True)
class SweepResult:
    records: list[ExperimentRecord]
    aggregated: list[AggregatedRecord]
    kappa_timing: list[KappaTimingRecord]


@dataclass(frozen=True)
class MatrixEvaluationResult:
    preconditioner: str
    solver: str
    kappa: float
    kappa_pre: float
    delta_kappa: float
    ratio_kappa: float
    converged: bool
    n_iter: int
    residual: float
    preconditioner_time: float
    kappa_eval_time: float
    solve_time: float
    total_time: float


class PreconditionerWorkbench:
    """Базовый конструктор с дефолтными компонентами

    - стандартные матрицы, солверы и предобуславливатели;
    - запуск sweep по размеру или по каппе;
    - предоставляет набор графиков по ним же;
    """
    def __init__(
        self,
        *,
        matrix_families: Optional[Sequence[MatrixFamily]] = None,
        preconditioners: Optional[Sequence[PreconditionerFactory]] = None,
        solvers: Optional[Sequence[SolverSpec]] = None,
    ) -> None:
        self.matrix_families: tuple[MatrixFamily, ...] = tuple(
            matrix_families if matrix_families is not None else default_matrix_families()
        )
        self.preconditioners: tuple[PreconditionerFactory, ...] = tuple(
            preconditioners if preconditioners is not None else default_preconditioners()
        )
        self.solvers: tuple[SolverSpec, ...] = tuple(
            solvers if solvers is not None else default_solvers(restart=40)
        )

    @staticmethod
    def default_sizes(
        *,
        min_size: int = 16,
        max_size: int = 512,
        count: int = 8,
    ) -> tuple[int, ...]:
        """Сформировать равномерную сетку размеров матриц"""
        return uniform_size_grid(min_size=min_size, max_size=max_size, count=count)

    def make_matrix(
        self,
        family_name: str,
        *,
        n: int,
        rng_seed: int = 0,
    ) -> Array:
        """Сгенерировать матрицу указанного семейства"""
        rng = np.random.default_rng(rng_seed)
        family = self._family_by_name(family_name)
        return family.generator(n, rng)

    def build_preconditioner(self, A: Array, name: str) -> Preconditioner:
        """Построить предобуславливатель по имени"""
        factory = self._preconditioner_by_name(name)
        return factory.build(A)

    def run_size_sweep(
        self,
        *,
        sizes: Optional[Iterable[int]] = None,
        n_samples: int = 6,
        family_count: Optional[int] = None,
        preconditioner_names: Optional[Sequence[str]] = None,
        solver_names: Optional[Sequence[str]] = None,
        rng_seed: int = 7,
        tol: float = 1e-8,
        maxiter: int = 400,
        signed_kappa: bool = True,
        kappa_bins: int = 12,
    ) -> SweepResult:
        """Запустить sweep по размеру матриц

        return:
            SweepResult с записями, агрегатами и временными оценками по каппе даи вообще все, что вносите
        """
        if sizes is None:
            sizes = self.default_sizes(min_size=16, max_size=512, count=8)
        size_tuple = tuple(int(n) for n in sizes)
        families = self._selected_families(family_count)
        preconditioners = tuple(self._selected_preconditioners(preconditioner_names))
        solvers = tuple(self._selected_solvers(solver_names))
        config = ExperimentConfig(
            sizes=size_tuple,
            n_samples=n_samples,
            matrix_families=tuple(families),
            preconditioners=preconditioners,
            solvers=solvers,
            rng_seed=rng_seed,
            tol=tol,
            maxiter=maxiter,
            signed_kappa=signed_kappa,
        )
        records = run_experiments(config)
        aggregated = aggregate_records(records)
        kappa_timing = aggregate_timing_by_kappa(records, bins=kappa_bins)
        return SweepResult(records=records, aggregated=aggregated, kappa_timing=kappa_timing)

    def run_kappa_sweep(
        self,
        *,
        matrix_size: int = 128,
        kappa_values: Sequence[float] = (-1e4, -1e3, -1e2, 1e2, 1e3, 1e4),
        n_samples: int = 4,
        preconditioner_names: Optional[Sequence[str]] = None,
        solver_names: Optional[Sequence[str]] = None,
        rng_seed: int = 11,
        tol: float = 1e-8,
        maxiter: int = 400,
        signed_kappa: bool = True,
        kappa_bins: int = 12,
    ) -> SweepResult:
        """Запустить sweep по каппе в каком-то спектре

        Для каждого значения kappa создается отдельное семейство матриц
            для дальнейших эксперементов по сетке мб
        """
        families = tuple(
            MatrixFamily(
                name=f"Random Spectrum ({kappa:+.1e})",
                generator=(lambda n, rng, kk=kappa: random_spectrum(n, rng, kappa=kk)),
                notes="Controlled signed spectral ratio",
            )
            for kappa in kappa_values
        )
        preconditioners = tuple(self._selected_preconditioners(preconditioner_names))
        solvers = tuple(self._selected_solvers(solver_names))
        config = ExperimentConfig(
            sizes=(int(matrix_size),),
            n_samples=n_samples,
            matrix_families=families,
            preconditioners=preconditioners,
            solvers=solvers,
            rng_seed=rng_seed,
            tol=tol,
            maxiter=maxiter,
            signed_kappa=signed_kappa,
        )
        records = run_experiments(config)
        aggregated = aggregate_records(records)
        kappa_timing = aggregate_timing_by_kappa(records, bins=kappa_bins)
        return SweepResult(records=records, aggregated=aggregated, kappa_timing=kappa_timing)

    def evaluate_matrix(
        self,
        A: Array,
        *,
        preconditioner: str = "Diagonal",
        solver: str = "GMRES",
        b: Optional[Vector] = None,
        tol: float = 1e-8,
        maxiter: int = 500,
        signed_kappa: bool = True,
        kappa_zero_tol: float = 1e-12,
    ) -> MatrixEvaluationResult:
        """Оценить одну матрицу и один предобуславливатель

        return:
            MatrixEvaluationResult с метриками kappa и временем работы
        """
        n = int(A.shape[0])
        if b is None:
            rng = np.random.default_rng(0)
            b = rng.standard_normal(n)
        factory = self._preconditioner_by_name(preconditioner)
        solver_spec = self._solver_by_name(solver)

        t_pre0 = perf_counter()
        precond = factory.build(A)
        preconditioner_time = perf_counter() - t_pre0

        kappa = condition_number(A, signed=signed_kappa, zero_tol=kappa_zero_tol)
        t_kappa0 = perf_counter()
        A_pre = precond.apply_to_matrix(A)
        kappa_pre = condition_number(A_pre, signed=signed_kappa, zero_tol=kappa_zero_tol)
        kappa_eval_time = perf_counter() - t_kappa0
        km = compute_kappa_metrics(kappa, kappa_pre)

        t_solve0 = perf_counter()
        result: SolverResult = solver_spec.solve(
            A, b, precond.as_linear_operator(), tol, maxiter
        )
        solve_time = perf_counter() - t_solve0
        total_time = preconditioner_time + kappa_eval_time + solve_time

        return MatrixEvaluationResult(
            preconditioner=preconditioner,
            solver=solver,
            kappa=km.kappa,
            kappa_pre=km.kappa_pre,
            delta_kappa=km.delta,
            ratio_kappa=km.ratio,
            converged=result.converged,
            n_iter=result.n_iter,
            residual=result.residual_norm,
            preconditioner_time=preconditioner_time,
            kappa_eval_time=kappa_eval_time,
            solve_time=solve_time,
            total_time=total_time,
        )

    def plot_family_bundle(
        self,
        sweep: SweepResult,
        *,
        matrix_family: str,
        solver: str = "GMRES",
        kappa_metric: str = "ratio",
        max_points_per_preconditioner: int = 200,
    ) -> dict[str, plt.Figure]:
        """Построить стандартный набор графиков для выбранного семейства"""
        fig1, _ = plot_delta_kappa_vs_n(
            sweep.aggregated,
            matrix_family=matrix_family,
            solver=solver,
            metric=kappa_metric,
        )
        fig2, _ = plot_iterations_vs_n(
            sweep.aggregated,
            matrix_family=matrix_family,
            solver=solver,
        )
        fig3, _ = plot_delta_kappa_vs_kappa(
            sweep.records,
            matrix_family=matrix_family,
            solver=solver,
            metric=kappa_metric,
            x_log=False,
            max_points_per_preconditioner=max_points_per_preconditioner,
        )
        fig4, _ = plot_time_vs_n(
            sweep.aggregated,
            matrix_family=matrix_family,
            solver=solver,
            metric="total",
        )
        fig5, _ = plot_time_vs_kappa(
            sweep.kappa_timing,
            matrix_family=matrix_family,
            solver=solver,
            metric="total",
            x_log=False,
            y_log=False,
        )
        return {
            "kappa_vs_n": fig1,
            "iterations_vs_n": fig2,
            "kappa_vs_kappa": fig3,
            "time_vs_n": fig4,
            "time_vs_kappa": fig5,
        }

    def _selected_families(self, family_count: Optional[int]) -> list[MatrixFamily]:
        if family_count is None:
            return list(self.matrix_families)
        return uniform_ordered_selection(self.matrix_families, family_count)

    def _family_by_name(self, name: str) -> MatrixFamily:
        for family in self.matrix_families:
            if family.name == name:
                return family
        raise ValueError(f"Unknown matrix family: {name!r}")

    def _selected_preconditioners(
        self, names: Optional[Sequence[str]]
    ) -> list[PreconditionerFactory]:
        if names is None:
            return list(self.preconditioners)
        selected = [self._preconditioner_by_name(name) for name in names]
        if not selected:
            raise ValueError("No preconditioners selected")
        return selected

    def _preconditioner_by_name(self, name: str) -> PreconditionerFactory:
        for precond in self.preconditioners:
            if precond.name == name:
                return precond
        raise ValueError(f"Unknown preconditioner: {name!r}")

    def _selected_solvers(self, names: Optional[Sequence[str]]) -> list[SolverSpec]:
        if names is None:
            return list(self.solvers)
        selected = [self._solver_by_name(name) for name in names]
        if not selected:
            raise ValueError("No solvers selected")
        return selected

    def _solver_by_name(self, name: str) -> SolverSpec:
        for solver in self.solvers:
            if solver.name == name:
                return solver
        raise ValueError(f"Unknown solver: {name!r}")


class PreconditionerWorkbenchV2:
    """Расширенный конструктор с реестрами
    functools:
        - регистрация компонентов по имени (матрицы, предобуславливатели, солверы);
        - возможность передавать имя, объект или функцию;
        - API для экспериментов и одиночной оценки
    """
    def __init__(
        self,
        *,
        matrix_families: Optional[Sequence[MatrixFamily]] = None,
        preconditioners: Optional[Sequence[PreconditionerFactory]] = None,
        solvers: Optional[Sequence[SolverSpec]] = None,
        register_defaults: bool = True,
    ) -> None:
        self._matrix_families: dict[str, MatrixFamily] = {}
        self._preconditioners: dict[str, PreconditionerFactory] = {}
        self._solvers: dict[str, SolverSpec] = {}
        self._family_order: list[str] = []
        self._precond_order: list[str] = []
        self._solver_order: list[str] = []

        if register_defaults:
            for fam in default_matrix_families():
                self.register_matrix_family(fam)
            for precond in default_preconditioners():
                self.register_preconditioner_factory(precond)
            for solver in default_solvers(restart=40):
                self.register_solver_spec(solver)

        if matrix_families is not None:
            for fam in matrix_families:
                self.register_matrix_family(fam)
        if preconditioners is not None:
            for precond in preconditioners:
                self.register_preconditioner_factory(precond)
        if solvers is not None:
            for solver in solvers:
                self.register_solver_spec(solver)

    def register_matrix_family(self, family: MatrixFamily) -> None:
        """Зарегистрировать семейство матриц"""
        if family.name not in self._matrix_families:
            self._family_order.append(family.name)
        self._matrix_families[family.name] = family

    def register_matrix_generator(
        self,
        name: str,
        generator: Callable[[int, np.random.Generator], Array],
        *,
        notes: str = "",
    ) -> None:
        """Упрощенная регистрация матрицы через генератор"""
        self.register_matrix_family(MatrixFamily(name=name, generator=generator, notes=notes))

    def register_preconditioner_factory(self, factory: PreconditionerFactory) -> None:
        """Зарегистрировать предобуславливателя"""
        if factory.name not in self._preconditioners:
            self._precond_order.append(factory.name)
        self._preconditioners[factory.name] = factory

    def register_solver_spec(self, solver: SolverSpec) -> None:
        """Зарегистрировать солвер"""
        if solver.name not in self._solvers:
            self._solver_order.append(solver.name)
        self._solvers[solver.name] = solver

    def register_preconditioner(
        self,
        name: str,
        builder: PreconditionerBuilder,
        *,
        max_n: Optional[int] = None,
        notes: str = "",
    ) -> None:
        """Зарегистрировать предобуславливатель по имени и функции сборки"""
        factory = PreconditionerFactory(name, builder, max_n=max_n, notes=notes)
        self.register_preconditioner_factory(factory)

    def register_solver(self, name: str, solve: SolverFn) -> None:
        """Зарегистрировать солвер по имени и функции solve"""
        self.register_solver_spec(SolverSpec(name, solve))

    def list_matrix_families(self) -> list[MatrixFamily]:
        """Вернуть список зарегистрированных семейств матриц"""
        return [self._matrix_families[name] for name in self._family_order]

    def list_preconditioners(self) -> list[PreconditionerFactory]:
        """Вернуть список зарегистрированных предобуславливателей"""
        return [self._preconditioners[name] for name in self._precond_order]

    def list_solvers(self) -> list[SolverSpec]:
        """Вернуть список зарегистрированных солверов"""
        return [self._solvers[name] for name in self._solver_order]

    def make_matrix(self, family: MatrixFamilyLike, *, n: int, rng_seed: int = 0) -> Array:
        """Сгенерировать матрицу по имени или объекту семейства"""
        rng = np.random.default_rng(rng_seed)
        fam = self._resolve_family(family)
        return fam.generator(n, rng)

    def build_preconditioner(self, A: Array,
                             preconditioner: PreconditionerLike) -> Preconditioner:
        """Построить предобуславливатель по имени или смотря какой fabric"""
        factory = self._resolve_preconditioner(preconditioner)
        return factory.build(A)

    def run_size_sweep(
        self,
        *,
        sizes: Optional[Iterable[int]] = None,
        n_samples: int = 4,
        matrix_families: Optional[Sequence[MatrixFamilyLike]] = None,
        preconditioners: Optional[Sequence[PreconditionerLike]] = None,
        solvers: Optional[Sequence[SolverLike]] = None,
        rng_seed: int = 7,
        tol: float = 1e-8,
        maxiter: int = 400,
        signed_kappa: bool = True,
        kappa_bins: int = 12,
    ) -> SweepResult:
        """Sweep по размеру с выбором компонентов;
        Можно передавать списки имен или объектов;

        ВАЖНО:
        Если список не указан то используются все зарегистрированные параметры
        """
        if sizes is None:
            sizes = uniform_size_grid(min_size=16, max_size=512, count=8)
        size_tuple = tuple(int(n) for n in sizes)
        families = self._resolve_families(matrix_families)
        preconds = self._resolve_preconditioners(preconditioners)
        solvers_resolved = self._resolve_solvers(solvers)
        config = ExperimentConfig(
            sizes=size_tuple,
            n_samples=n_samples,
            matrix_families=tuple(families),
            preconditioners=tuple(preconds),
            solvers=tuple(solvers_resolved),
            rng_seed=rng_seed,
            tol=tol,
            maxiter=maxiter,
            signed_kappa=signed_kappa,
        )
        records = run_experiments(config)
        aggregated = aggregate_records(records)
        kappa_timing = aggregate_timing_by_kappa(records, bins=kappa_bins)
        return SweepResult(records=records, aggregated=aggregated, kappa_timing=kappa_timing)

    def evaluate_matrix(
        self,
        A: Array,
        *,
        preconditioner: Union[Preconditioner, PreconditionerLike, PreconditionerBuilder],
        solver: Union[SolverLike, SolverFn],
        b: Optional[Vector] = None,
        tol: float = 1e-8,
        maxiter: int = 500,
        signed_kappa: bool = True,
        kappa_zero_tol: float = 1e-12,
    ) -> MatrixEvaluationResult:
        """Оценить одну матрицу с пользовательскими компонентами"""
        n = int(A.shape[0])
        if b is None:
            rng = np.random.default_rng(0)
            b = rng.standard_normal(n)

        precond, preconditioner_time = self._build_preconditioner(A, preconditioner)
        solver_spec = self._resolve_solver_instance(solver)

        kappa = condition_number(A, signed=signed_kappa, zero_tol=kappa_zero_tol)
        t_kappa0 = perf_counter()
        A_pre = precond.apply_to_matrix(A)
        kappa_pre = condition_number(A_pre, signed=signed_kappa, zero_tol=kappa_zero_tol)
        kappa_eval_time = perf_counter() - t_kappa0
        km = compute_kappa_metrics(kappa, kappa_pre)

        t_solve0 = perf_counter()
        result: SolverResult = solver_spec.solve(
            A, b, precond.as_linear_operator(), tol, maxiter
        )
        solve_time = perf_counter() - t_solve0
        total_time = preconditioner_time + kappa_eval_time + solve_time

        return MatrixEvaluationResult(
            preconditioner=precond.name,
            solver=solver_spec.name,
            kappa=km.kappa,
            kappa_pre=km.kappa_pre,
            delta_kappa=km.delta,
            ratio_kappa=km.ratio,
            converged=result.converged,
            n_iter=result.n_iter,
            residual=result.residual_norm,
            preconditioner_time=preconditioner_time,
            kappa_eval_time=kappa_eval_time,
            solve_time=solve_time,
            total_time=total_time,
        )

    def _resolve_family(self, family: MatrixFamilyLike) -> MatrixFamily:
        if isinstance(family, MatrixFamily):
            return family
        if family in self._matrix_families:
            return self._matrix_families[family]
        raise ValueError(f"Unknown matrix family: {family!r}")

    def _resolve_families(
        self, families: Optional[Sequence[MatrixFamilyLike]]
    ) -> list[MatrixFamily]:
        if families is None:
            return self.list_matrix_families()
        return [self._resolve_family(fam) for fam in families]

    def _resolve_preconditioner(self, precond: PreconditionerLike) -> PreconditionerFactory:
        if isinstance(precond, PreconditionerFactory):
            return precond
        if precond in self._preconditioners:
            return self._preconditioners[precond]
        raise ValueError(f"Unknown preconditioner: {precond!r}")

    def _resolve_preconditioners(
        self, preconds: Optional[Sequence[PreconditionerLike]]
    ) -> list[PreconditionerFactory]:
        if preconds is None:
            return self.list_preconditioners()
        return [self._resolve_preconditioner(p) for p in preconds]

    def _resolve_solver(self, solver: SolverLike) -> SolverSpec:
        if isinstance(solver, SolverSpec):
            return solver
        if solver in self._solvers:
            return self._solvers[solver]
        raise ValueError(f"Unknown solver: {solver!r}")

    def _resolve_solvers(self, solvers: Optional[Sequence[SolverLike]]) -> list[SolverSpec]:
        if solvers is None:
            return self.list_solvers()
        return [self._resolve_solver(s) for s in solvers]

    def _build_preconditioner(
        self,
        A: Array,
        precond: Union[Preconditioner,
                       PreconditionerLike,
                       PreconditionerBuilder],
    ) -> tuple[Preconditioner, float]:
        if isinstance(precond, Preconditioner):
            return precond, 0.0
        t0 = perf_counter()
        if isinstance(precond, PreconditionerFactory) or isinstance(precond,
                                                                    str):
            built = self._resolve_preconditioner(precond).build(A)
        else:
            built = precond(A)
        return built, perf_counter() - t0

    def _resolve_solver_instance(self,
                                 solver: Union[SolverLike, SolverFn]) -> SolverSpec:
        if isinstance(solver, SolverSpec):
            return solver
        if isinstance(solver, str):
            return self._resolve_solver(solver)
        return SolverSpec("Custom", solver)
