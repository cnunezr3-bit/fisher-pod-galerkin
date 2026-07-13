# fom.py
from __future__ import annotations
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def cost_function(m_int: np.ndarray, u: np.ndarray, Y: np.ndarray, yD: np.ndarray, dt: float) -> float:
    # 0.5 (y(T)-yD)^T M (y(T)-yD) + 0.5 sum dt u^T M u
    diff = Y[-1] - yD
    term_final = 0.5 * float(np.sum(m_int * (diff * diff)))
    term_u = 0.5 * dt * float(np.sum(m_int[None, :] * (u[1:] * u[1:])))
    return term_final + term_u

def solve_state_fem_lumped(mu: float, dt: float, Nt: int,
                           m_int: np.ndarray, A_int: sp.csr_matrix,
                           u_seq: np.ndarray, y0: np.ndarray,
                           newton_tol=1e-8, newton_maxit=20):
    """
    Implicit Euler + Newton with FEM P1 + lumped mass:

      M (y_{k+1}-y_k) + dt[ mu A y_{k+1}
                           + M (u_{k+1} ⊙ y_{k+1})
                           - M y_{k+1}
                           + M (y_{k+1} ⊙ y_{k+1}) ] = 0

    where M is diagonal with entries m_int.
    """
    n = m_int.size
    assert u_seq.shape == (Nt + 1, n)
    Mdiag = m_int
    Y = np.zeros((Nt + 1, n), dtype=float)
    Y[0] = y0.copy()
    newton_iters = []
    for k in range(Nt):
        uk1 = u_seq[k + 1]
        yk = Y[k]
        w = yk.copy()
        it_used = 0
        for it in range(1, newton_maxit + 1):
            it_used = it
            Mw_minus = Mdiag * (w - yk)
            Aw = A_int @ w
            F = Mw_minus + dt * (mu * Aw + Mdiag * (uk1 * w) - Mdiag * w + Mdiag * (w * w))
            if np.linalg.norm(F) / np.sqrt(n) < newton_tol:
                break
            diagJ = Mdiag + dt * (Mdiag * uk1 - Mdiag + 2.0 * Mdiag * w)
            J = sp.diags(diagJ, 0, format="csr") + (dt * mu) * A_int
            dw = spla.spsolve(J, -F)
            w = w + dw
            if np.linalg.norm(dw) / (np.linalg.norm(w) + 1e-14) < 1e-10:
                break
        newton_iters.append(it_used)
        Y[k + 1] = w
    return Y, newton_iters
def solve_adjoint_fem_lumped(mu: float, dt: float, Nt: int,
                             m_int: np.ndarray, A_int: sp.csr_matrix,
                             u_seq: np.ndarray, Y: np.ndarray, yD: np.ndarray):
    """
    Adjoint (lumped mass):
      -p_t - mu Δp + (u - (1-2y)) p = 0
    Weak semi-discrete:
      M(-p_dot) + mu A p + M( diagterm ⊙ p ) = 0
    Backward implicit stepping:
      (M + dt( mu A + diag(M*diagterm_k) )) p_k = M p_{k+1}
    Terminal:
      p(T) = y(T) - yD  (nodal, consistent with L2 via M)
    """
    n = m_int.size
    Mdiag = m_int
    P = np.zeros((Nt + 1, n), dtype=float)
    P[Nt] = Y[Nt] - yD
    Mmat = sp.diags(Mdiag, 0, format="csr")
    for k in range(Nt - 1, -1, -1):
        uk = u_seq[k]
        yk = Y[k]
        diagterm = uk - (1.0 - 2.0 * yk)  
        A = Mmat + dt * ((mu) * A_int + sp.diags(Mdiag * diagterm, 0, format="csr"))
        rhs = Mdiag * P[k + 1]
        P[k] = spla.spsolve(A, rhs)
    return P
def posteriori_indicator_xi(u: np.ndarray, Y: np.ndarray, P: np.ndarray):
    """
    Same structure as your original indicator, pointwise.
    """
    s = u - (Y * P)
    xi = np.zeros_like(u)
    m0 = (u <= 1e-14)
    m1 = (u >= 1.0 - 1e-14)
    mi = (~m0) & (~m1)
    xi[m0] = np.minimum(s[m0], 0.0)
    xi[mi] = -s[mi]
    xi[m1] = -np.maximum(s[m1], 0.0)
    return xi
def pdas_fom_fem(mu: float, T: float, dt: float,
                 m_int: np.ndarray, A_int: sp.csr_matrix,
                 y0: np.ndarray, yD: np.ndarray,
                 maxit=30, tol=1e-8, verbose=True):
    Nt = int(np.round(T / dt))
    n = m_int.size
    times = np.linspace(0.0, T, Nt + 1)
    u = np.zeros((Nt + 1, n), dtype=float)
    t0 = time.perf_counter()
    pdas_iters = 0
    newton_iters_last = None
    for it in range(1, maxit + 1):
        pdas_iters = it
        Y, newton_iters = solve_state_fem_lumped(mu, dt, Nt, m_int, A_int, u, y0)
        newton_iters_last = newton_iters
        P = solve_adjoint_fem_lumped(mu, dt, Nt, m_int, A_int, u, Y, yD)
        u_new = np.clip(Y * P, 0.0, 1.0)
        diff = np.linalg.norm(u_new - u) / (np.linalg.norm(u_new) + 1e-12)
        u = u_new
        if verbose:
            J = cost_function(m_int, u, Y, yD, dt)
            print(f"[FOM(FEM) PDAS] it={it:02d}  rel_change(u)={diff:.3e}  J={J:.6e}")
        if diff < tol:
            break
    t1 = time.perf_counter()
    avg_newton = float(np.mean(newton_iters_last)) if newton_iters_last is not None else float("nan")
    return {
        "u": u, "Y": Y, "P": P,
        "Nt": Nt, "times": times,
        "runtime": t1 - t0,
        "pdas_iters": pdas_iters,
        "avg_newton_per_step": avg_newton,
        "newton_iters_per_step": newton_iters_last,
    }

