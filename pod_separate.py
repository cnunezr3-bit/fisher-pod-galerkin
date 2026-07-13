import numpy as np

def build_separate_pod_basis_lumped(m_int, Y, P, d_snap, l, dt, time_weight=True):

    Nt = Y.shape[0] - 1
    n = Y.shape[1]

    d_snap = min(d_snap, Nt)
    snap_idx = np.unique(np.round(np.linspace(1, Nt, d_snap)).astype(int))
    m = len(snap_idx)

    if time_weight:
        Wt = dt * np.ones(m)
    else:
        Wt = np.ones(m)

 
    Zy = np.zeros((n, m))
    for j, k in enumerate(snap_idx):
        Zy[:, j] = Y[k]

    Zy = Zy * np.sqrt(Wt)[None, :]
    MZy = Zy * m_int[:, None]
    Cy = Zy.T @ MZy

    evals_y, evecs_y = np.linalg.eigh(Cy)
    idx = np.argsort(evals_y)[::-1]
    evals_y = np.maximum(evals_y[idx], 0.0)
    evecs_y = evecs_y[:, idx]

    l_eff = min(l, np.sum(evals_y > 1e-14))
    Vy = evecs_y[:, :l_eff]
    lam_y = evals_y[:l_eff]

    Psi_y = Zy @ (Vy / np.sqrt(lam_y + 1e-30))


    Zp = np.zeros((n, m))
    for j, k in enumerate(snap_idx):
        Zp[:, j] = P[k]

    Zp = Zp * np.sqrt(Wt)[None, :]
    MZp = Zp * m_int[:, None]
    Cp = Zp.T @ MZp

    evals_p, evecs_p = np.linalg.eigh(Cp)
    idx = np.argsort(evals_p)[::-1]
    evals_p = np.maximum(evals_p[idx], 0.0)
    evecs_p = evecs_p[:, idx]

    l_eff = min(l, np.sum(evals_p > 1e-14))
    Vp = evecs_p[:, :l_eff]
    lam_p = evals_p[:l_eff]

    Psi_p = Zp @ (Vp / np.sqrt(lam_p + 1e-30))

    return Psi_y, Psi_p, evals_y, evals_p, snap_idx

