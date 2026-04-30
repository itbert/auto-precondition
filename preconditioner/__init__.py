from .matrices import (
    MatrixFamily,
    default_matrix_families,
    generate_matrices,
    uniform_ordered_selection,
    uniform_size_grid,
)
from .preconditioners import (
    Preconditioner,
    PreconditionerFactory,
    default_preconditioners,
)
from .solvers import (
    SolverResult,
    SolverSpec,
    default_solvers,
)
from .experiments import (
    KappaTimingRecord,
    ExperimentConfig,
    ExperimentRecord,
    AggregatedRecord,
    run_experiments,
    aggregate_records,
    aggregate_timing_by_kappa
)
from .metrics import condition_number, compute_kappa_metrics
from .plotting import (
    apply_plot_style,
    plot_delta_kappa_vs_n,
    plot_delta_kappa_vs_kappa,
    plot_iterations_vs_n,
    plot_time_vs_kappa,
    plot_time_vs_n,
)
from .workbench import (
    MatrixEvaluationResult,
    PreconditionerWorkbench,
    PreconditionerWorkbenchV2,
    SweepResult,
)
from .ml import (
    InverseApproximationMetrics,
    RidgeInverseApproximator,
    build_inverse_dataset,
    evaluate_inverse_model,
)
from .cnn import (
    CNNInverseApproximatorConfig,
    CNNTrainingHistory,
    ConvolutionalInverseApproximator,
    build_cnn_preconditioner_factory,
    evaluate_cnn_inverse_model,
    torch_is_available,
)

__all__ = [
    "MatrixFamily",
    "default_matrix_families",
    "generate_matrices",
    "uniform_ordered_selection",
    "uniform_size_grid",
    "Preconditioner",
    "PreconditionerFactory",
    "default_preconditioners",
    "SolverResult",
    "SolverSpec",
    "default_solvers",
    "ExperimentConfig",
    "ExperimentRecord",
    "AggregatedRecord",
    "KappaTimingRecord",
    "run_experiments",
    "aggregate_records",
    "aggregate_timing_by_kappa",
    "condition_number",
    "compute_kappa_metrics",
    "apply_plot_style",
    "plot_delta_kappa_vs_n",
    "plot_delta_kappa_vs_kappa",
    "plot_iterations_vs_n",
    "plot_time_vs_n",
    "plot_time_vs_kappa",
    "SweepResult",
    "MatrixEvaluationResult",
    "PreconditionerWorkbench",
    "PreconditionerWorkbenchV2",
    "RidgeInverseApproximator",
    "InverseApproximationMetrics",
    "build_inverse_dataset",
    "evaluate_inverse_model",
    "CNNInverseApproximatorConfig",
    "CNNTrainingHistory",
    "ConvolutionalInverseApproximator",
    "build_cnn_preconditioner_factory",
    "evaluate_cnn_inverse_model",
    "torch_is_available",
]
