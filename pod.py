# pod.py
from __future__ import annotations
import numpy as np

def build_coupled_pod_basis_lumped(m_int: np.ndarray,
                                   Y: np.ndarray, P: np.ndarray,
                                   d_snap: int, l: int,
                                   dt: float, time_weight: bool = True):
    """
    Coupled POD for z=(y,p) in R^{2n} with inner product:
      <z1,z2> = y1^T M y2 + p1^T M p2
    where M is diagonal with entries m_int (lumped).

    We form correlation matrix:
      C = Zc^T (M2 Zc),
    where Zc = Z * sqrt(Wt) (column scaling),
          M2 = blockdiag(M, M), applied via m_int.

    Then modes:
      Psi2 = Zc V / sqrt(lam)
    so that Psi2^T M2 Psi2 = I.
    """
    Nt = Y.shape[0] - 1
    n = Y.shape[1]
    assert m_int.size == n
    d_snap = min(d_snap, Nt)
    snap_idx = np.unique(np.round(np.linspace(1, Nt, d_snap)).astype(int))
    m = len(snap_idx)

    Z = np.zeros((2 * n, m), dtype=float)
    for j, k in enumerate(snap_idx):
        Z[:n, j] = Y[k]
        Z[n:, j] = P[k]
    if time_weight:
        Wt = dt * np.ones(m, dtype=float)
    else:
        Wt = np.ones(m, dtype=float)
    Zc = Z * np.sqrt(Wt)[None, :]
    MZc = Zc.copy()
    MZc[:n, :] *= m_int[:, None]
    MZc[n:, :] *= m_int[:, None]
    C = Zc.T @ MZc  
    evals, evecs = np.linalg.eigh(C)
    idx = np.argsort(evals)[::-1]
    evals = np.maximum(evals[idx], 0.0)
    evecs = evecs[:, idx]
    r = int(np.sum(evals > 1e-14))
    l_eff = min(l, r) if r > 0 else 0
    if l_eff == 0:
        raise RuntimeError("POD: no positive eigenvalues; check snapshots.")
    V = evecs[:, :l_eff]
    lam = evals[:l_eff]
    Psi2 = Zc @ (V / np.sqrt(lam + 1e-30))
    Psi_y = Psi2[:n, :]
    Psi_p = Psi2[n:, :]
    return Psi_y, Psi_p, Psi2, evals, snap_idx

