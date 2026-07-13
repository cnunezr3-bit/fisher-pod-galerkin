from __future__ import annotations
import numpy as np

from config import T_FINAL, DT, NX_COARSE, D_SNAP, TAB_DIR
from fem_mesh import build_structured_tri_mesh
from fem_operators import assemble_mass_lumped_and_stiffness_interior
from fom import pdas_fom_fem
from pod import build_coupled_pod_basis_lumped
from rom import pdas_rom_coupled_fem
from io_utils import to_sci, write_latex_table


def l2_norm_lumped(m_int: np.ndarray, v: np.ndarray) -> float:
    return float(np.sqrt(np.sum(m_int * (v * v))))

def l2_error_lumped(m_int: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
    return l2_norm_lumped(m_int, v1 - v2)

def rel_l2_error_lumped(m_int: np.ndarray, v_ref: np.ndarray, v_app: np.ndarray) -> float:
    return l2_error_lumped(m_int, v_ref, v_app) / (l2_norm_lumped(m_int, v_ref) + 1e-14)

def build_y0_on_interior(mesh):
    y0 = np.zeros(mesh.n_int, dtype=float)
    for idx_int, g in enumerate(mesh.interior):
        x, y = mesh.nodes[g]
        y0[idx_int] = 16.0 * x * y * (x - 1.0) * (y - 1.0)
    return y0

def run_out_of_training_test():
    gamma_train = 1.0e-3
    gamma_test = 2.0e-3
    T = T_FINAL
    dt = DT
    l_list = [3, 5, 10]
    mesh = build_structured_tri_mesh(NX_COARSE)
    m_c, A_c = assemble_mass_lumped_and_stiffness_interior(mesh)
    y0 = build_y0_on_interior(mesh)
    yD = 0.5 * np.ones(mesh.n_int, dtype=float)
    print("\n==========================================")
    print(" Out-of-training parameter robustness test")
    print("==========================================")
    print(f"Training gamma: {gamma_train:.2e}")
    print(f"Testing gamma:  {gamma_test:.2e}")
    print("\n--- Solving training FOM ---")
    res_train = pdas_fom_fem(gamma_train, T, dt, m_c, A_c, y0, yD,maxit=25, tol=1e-8, verbose=True)
    l_max = max(l_list)
    Psi_y_max, Psi_p_max, _, evals, snap_idx = build_coupled_pod_basis_lumped(
        m_c,
        res_train["Y"],
        res_train["P"],
        d_snap=D_SNAP,
        l=l_max,dt=dt,time_weight=True)
    print(f"\nPOD basis built using gamma = {gamma_train:.2e}")
    print(f"Snapshots used: {len(snap_idx)}")
    print(f"Top eigenvalues: {evals[:min(5, len(evals))]}")
    print("\n--- Solving test FOM reference ---")
    res_test_fom = pdas_fom_fem(
        gamma_test, T, dt, m_c, A_c, y0, yD,
        maxit=25, tol=1e-8, verbose=True)
    rows = []
    for l in l_list:
        print(f"\n--- Testing ROM with l={l} ---")
        Psi_y = Psi_y_max[:, :l]
        Psi_p = Psi_p_max[:, :l]
        res_rom = pdas_rom_coupled_fem(
            gamma_test, T, dt, m_c, A_c,
            y0, yD,
            Psi_y, Psi_p,
            maxit=25, tol=1e-6, verbose=True)
        ey = l2_error_lumped(m_c, res_test_fom["Y"][-1], res_rom["Y"][-1])
        ep = l2_error_lumped(m_c, res_test_fom["P"][-1], res_rom["P"][-1])
        Eu = rel_l2_error_lumped(m_c, res_test_fom["u"][-1], res_rom["u"][-1])
        rows.append([str(l),to_sci(ey, 3),to_sci(ep, 3),to_sci(Eu, 3),])
    write_latex_table(
        TAB_DIR / "table_out_of_training_gamma.tex",
        caption=("Out-of-training parameter robustness test. " "The POD basis is constructed with $\\gamma=10^{-3}$ " "and tested with $\\gamma=2\\times 10^{-3}$."), label="tab:out_of_training_gamma", headers=[r"$l$",r"$e_y$",r"$e_p$",r"$E_u$",], rows=rows, align="c|ccc")
    print("\n--- Output written ---")
    print((TAB_DIR / "table_out_of_training_gamma.tex").resolve())

if __name__ == "__main__":
    run_out_of_training_test()
