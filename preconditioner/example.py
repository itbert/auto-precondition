import numpy as np
from preconditioner import (Preconditioner,
                            PreconditionerWorkbenchV2,
                            SolverResult)

wb = PreconditionerWorkbenchV2(register_defaults=False)

# custom matrix family
wb.register_matrix_generator(
    "...",
    lambda n, rng: rng.standard_normal((n, n)) + 2.0 * np.eye(n),
    notes="Dense + shift"
)

# custom preconditioner
def my_builder(A):
    n = A.shape[0]
    return Preconditioner("...", lambda x: x / np.diag(A), n)

wb.register_preconditioner("...", my_builder)

# custom solver (callable)
def my_solver(A, b, M, tol, maxiter):
    # ... solver какой-нибудь ...
    return SolverResult(converged=True, n_iter=5, residual_norm=1e-9)

wb.register_solver("...", my_solver)

A = wb.make_matrix("...", n=64)
res = wb.evaluate_matrix(A, preconditioner="...", solver="...")
