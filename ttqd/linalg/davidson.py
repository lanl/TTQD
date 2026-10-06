import numpy as np


def _orthonormalize_against(vec, basis, tol=1.0e-12):
    """Modified Gram-Schmidt against an existing orthonormal basis."""
    for _ in range(2):
        for base in basis:
            vec = vec - base * np.vdot(base, vec)

    norm = np.linalg.norm(vec)
    if norm <= tol:
        return vec, norm
    return vec / norm, norm


def make_diagonal_preconditioner(diagonal, min_denom=1.0e-12):
    r"""Build a Davidson diagonal preconditioner.

    The returned function has signature ``preconditioner(residual, theta)`` and
    applies ``residual / (theta - diagonal)`` with a small denominator guard.
    """
    diag = np.real(np.asarray(diagonal).reshape(-1))

    def preconditioner(residual, theta):
        denom = theta - diag
        small = np.abs(denom) < min_denom
        if np.any(small):
            signs = np.where(denom[small] >= 0.0, 1.0, -1.0)
            denom = denom.copy()
            denom[small] = signs * min_denom
        return residual / denom

    return preconditioner


def davidson_lowest_eigenpair(
    matvec,
    guess,
    preconditioner=None,
    size=None,
    dtype=None,
    tol=1.0e-8,
    maxiter=80,
    max_subspace=24,
):
    r"""Find the lowest eigenpair with a single-root Davidson iteration.

    Parameters
    ----------
    matvec
        Callable implementing ``H @ x`` for a flat vector ``x``.
    guess
        Initial guess vector.
    preconditioner
        Optional callable ``preconditioner(residual, theta)`` returning the
        correction vector to add to the Davidson subspace.
    size
        Flat problem dimension. Defaults to ``guess.size``.
    dtype
        Vector dtype. Defaults to the dtype inferred from ``guess``.
    tol
        Relative residual tolerance.
    maxiter
        Maximum Davidson correction iterations.
    max_subspace
        Maximum subspace dimension before restarting with the Ritz vector plus
        the latest correction.

    Returns
    -------
    energy, vector, stats
        Lowest Ritz value, corresponding flat vector, and a small diagnostics
        dictionary.
    """
    guess = np.asarray(guess)
    if size is None:
        size = guess.size
    size = int(size)
    max_subspace = max(2, int(max_subspace))
    maxiter = max(1, int(maxiter))
    tol = 0.0 if tol is None else float(tol)
    if dtype is None:
        dtype = np.result_type(guess)

    v0 = np.asarray(guess, dtype=dtype).reshape(size)
    nrm = np.linalg.norm(v0)
    if nrm <= 1.0e-14:
        v0 = np.zeros(size, dtype=dtype)
        v0[0] = 1.0
    else:
        v0 = v0 / nrm

    V = [v0]
    AV = [matvec(v0)]
    best_energy = None
    best_vector = v0
    best_residual = np.inf
    iteration = 0

    for iteration in range(1, maxiter + 1):
        vmat = np.column_stack(V)
        avmat = np.column_stack(AV)
        hsub = vmat.conj().T @ avmat
        hsub = 0.5 * (hsub + hsub.conj().T)

        evals, evecs = np.linalg.eigh(hsub)
        energy = float(np.real(evals[0]))
        coeff = evecs[:, 0]
        vector = vmat @ coeff
        hvector = avmat @ coeff

        vnorm = np.linalg.norm(vector)
        if vnorm > 0.0:
            vector = vector / vnorm
            hvector = hvector / vnorm
            energy = float(np.real(np.vdot(vector, hvector)))

        residual = hvector - energy * vector
        residual_norm = float(np.linalg.norm(residual))
        best_energy = energy
        best_vector = vector
        best_residual = residual_norm

        if residual_norm <= tol * max(1.0, abs(energy)):
            return energy, vector, {
                "converged": True,
                "iterations": iteration,
                "residual_norm": residual_norm,
            }

        if preconditioner is None:
            correction = residual.copy()
        else:
            correction = preconditioner(residual, energy)

        correction, correction_norm = _orthonormalize_against(correction, V)
        if correction_norm <= 1.0e-12:
            correction, correction_norm = _orthonormalize_against(residual.copy(), V)
        if correction_norm <= 1.0e-12:
            break

        hcorrection = matvec(correction)
        if len(V) >= max_subspace:
            V = [vector, correction]
            AV = [hvector, hcorrection]
        else:
            V.append(correction)
            AV.append(hcorrection)

    return best_energy, best_vector, {
        "converged": False,
        "iterations": iteration,
        "residual_norm": best_residual,
    }
