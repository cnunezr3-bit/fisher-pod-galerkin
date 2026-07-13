# rom.py
from __future__ import annotations
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from fom import cost_function

def _My_Ay(m_int: np.ndarray, A_int: sp.csr_matrix, Psi: np.ndarray):
    # My = Psi^T M Psi, Ay = Psi^T A Psi, with lumped M
    MPsi = (m_int[:, None] * Psi)
    My = Psi.T @ MPsi
    Ay = Psi.T @ (A_int @ Psi)
    return My, Ay

def solve_state_rom_fem_lumped(mu: float, dt: float, Nt: int,
                               m_int: np.ndarray, A_int: sp.csr_matrix,
                               u_seq: np.ndarray, y0: np.ndarray,
                               Psi_y: np.ndarray,
                               newton_tol=1e-10, newton_maxit=30):
    """
    ROM for state with Galerkin testing using Psi_y:

    Full residual:
      R(y) = M(y - yk) + dt( mu A y + M(u⊙y) - M y + M(y⊙y) )
    Reduced residual:
      Fr(a) = Psi^T R(Psi a)

    Newton in reduced coordinates a.
    """
    n, l = Psi_y.shape
    assert m_int.size == n
    My, Ay = _My_Ay(m_int, A_int, Psi_y)
    rhs0 = Psi_y.T @ (m_int * y0)
    a0 = np.linalg.solve(My, rhs0)

    Acoef = np.zeros((Nt + 1, l), dtype=float)
    Yfull = np.zeros((Nt + 1, n), dtype=float)
    Acoef[0] = a0
    Yfull[0] = Psi_y @ a0
    newton_iters = []
    for k in range(Nt):
        uk1 = u_seq[k + 1]
        ak = Acoef[k].copy()
        w = ak.copy()
        it_used = 0
        for it in range(1, newton_maxit + 1):
            it_used = it
            y = Psi_y @ w
            yk = Yfull[k]
            R = (m_int * (y - yk)) + dt * (mu * (A_int @ y) + m_int * (uk1 * y) - m_int * y + m_int * (y * y))
            Fr = Psi_y.T @ R
            if np.linalg.norm(Fr) / np.sqrt(l) < newton_tol:
                break
            term_diag = (m_int * (uk1 - 1.0 + 2.0 * y))[:, None] * Psi_y  # m*(u-1+2y)*Psi
            JPsi = (m_int[:, None] * Psi_y) + dt * (mu * (A_int @ Psi_y) + term_diag)
            Jr = Psi_y.T @ JPsi
            dw = np.linalg.solve(Jr, -Fr)
            w = w + dw
            if np.linalg.norm(dw) / (np.linalg.norm(w) + 1e-14) < 1e-12:
                break
        newton_iters.append(it_used)
        Acoef[k + 1] = w
        Yfull[k + 1] = Psi_y @ w

    return Acoef, Yfull, newton_iters

def solve_adjoint_rom_fem_lumped(mu: float, dt: float, Nt: int,
                                 m_int: np.ndarray, A_int: sp.csr_matrix,
                                 u_seq: np.ndarray, Yrom: np.ndarray, yD: np.ndarray,
                                 Psi_p: np.ndarray):
    """
    ROM for adjoint:
      (M + dt(mu A + diag(m*diagterm_k))) p_k = M p_{k+1}
    with p ≈ Psi_p b.
    """
    n, l = Psi_p.shape
    assert m_int.size == n
    Mp, Ap = _My_Ay(m_int, A_int, Psi_p)
    B = np.zeros((Nt + 1, l), dtype=float)
    Pfull = np.zeros((Nt + 1, n), dtype=float)
    pT = Yrom[Nt] - yD
    rhsT = Psi_p.T @ (m_int * pT)
    bT = np.linalg.solve(Mp, rhsT)
    B[Nt] = bT
    Pfull[Nt] = Psi_p @ bT
    for k in range(Nt - 1, -1, -1):
        uk = u_seq[k]
        yk = Yrom[k]
        diagterm = uk - (1.0 - 2.0 * yk)
        Dk = Psi_p.T @ ((m_int * diagterm)[:, None] * Psi_p)
        Ar = Mp + dt * (mu * Ap + Dk)
        rhs = Mp @ B[k + 1]  
        bk = np.linalg.solve(Ar, rhs)
        B[k] = bk
        Pfull[k] = Psi_p @ bk
    return B, Pfull
def pdas_rom_coupled_fem(mu: float, T: float, dt: float,
                         m_int: np.ndarray, A_int: sp.csr_matrix,
                         y0: np.ndarray, yD: np.ndarray,
                         Psi_y: np.ndarray, Psi_p: np.ndarray,
                         maxit=30, tol=1e-6, verbose=True):
    Nt = int(np.round(T / dt))
    n = m_int.size
    times = np.linspace(0.0, T, Nt + 1)
    u = np.zeros((Nt + 1, n), dtype=float)
    t0 = time.perf_counter()
    pdas_iters = 0
    newton_iters_last = None
    for it in range(1, maxit + 1):
        pdas_iters = it
        _, Yrom, newton_iters = solve_state_rom_fem_lumped(mu, dt, Nt, m_int, A_int, u, y0, Psi_y)
        newton_iters_last = newton_iters
        _, Prom = solve_adjoint_rom_fem_lumped(mu, dt, Nt, m_int, A_int, u, Yrom, yD, Psi_p)
        u_new = np.clip(Yrom * Prom, 0.0, 1.0)
        diff = np.linalg.norm(u_new - u) / (np.linalg.norm(u_new) + 1e-12)
        u = u_new
        if verbose:
            J = cost_function(m_int, u, Yrom, yD, dt)
            print(f"[ROM(FEM) PDAS] it={it:02d}  rel_change(u)={diff:.3e}  J={J:.6e}")
        if diff < tol:
            break
    t1 = time.perf_counter()
    avg_newton = float(np.mean(newton_iters_last)) if newton_iters_last is not None else float("nan")

    return {
        "u": u, "Y": Yrom, "P": Prom,
        "Nt": Nt, "times": times,
        "runtime": t1 - t0,
        "pdas_iters": pdas_iters,
        "avg_newton_per_step": avg_newton,
        "newton_iters_per_step": newton_iters_last,
    }

