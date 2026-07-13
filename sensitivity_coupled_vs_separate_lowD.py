from __future__ import annotations
import numpy as np
from config import MU, T_FINAL, DT, NX_COARSE, L_LIST, TAB_DIR
from fem_mesh import build_structured_tri_mesh
from fem_operators import assemble_mass_lumped_and_stiffness_interior
from fom import pdas_fom_fem, cost_function
from pod import build_coupled_pod_basis_lumped
from pod_separate import build_separate_pod_basis_lumped
from rom import pdas_rom_coupled_fem
from io_utils import to_sci, write_latex_table
def l2_norm_lumped(m_int: np.ndarray, v: np.ndarray) -> float:
    return float(np.sqrt(np.sum(m_int * (v * v))))
def rel_l2_error_lumped(m_int: np.ndarray, v_ref: np.ndarray, v_app: np.ndarray) -> float:
    num = l2_norm_lumped(m_int, v_ref - v_app)
    den = l2_norm_lumped(m_int, v_ref) + 1e-14
    return num / den
def build_y0_on_interior(mesh):
    y0 = np.zeros(mesh.n_int, dtype=float)
    for idx_int, g in enumerate(mesh.interior):
        x, y = mesh.nodes[g]
        y0[idx_int] = 16.0 * x * y * (x - 1.0) * (y - 1.0)
    return y0
def run():
    gamma = MU
    T = T_FINAL
    dt = DT
    DS = [3, 5]
    mesh = build_structured_tri_mesh(NX_COARSE)
    m_c, A_c = assemble_mass_lumped_and_stiffness_interior(mesh)
    y0 = build_y0_on_interior(mesh)
    yD = 0.5 * np.ones(mesh.n_int, dtype=float)
    res_fom = pdas_fom_fem(gamma, T, dt, m_c, A_c, y0, yD, maxit=25, tol=1e-8, verbose=True)
    J_fom = cost_function(m_c, res_fom["u"], res_fom["Y"], yD, dt)
    uT_fom = res_fom["u"][-1]
    rows = []
    for d in DS:
        for l in L_LIST:
            Psi_y_c, Psi_p_c, _, _, _ = build_coupled_pod_basis_lumped(m_c, res_fom["Y"], res_fom["P"], d_snap=d, l=l, dt=dt, time_weight=True)
            res_c = pdas_rom_coupled_fem(gamma, T, dt, m_c, A_c, y0, yD, Psi_y_c, Psi_p_c,maxit=25, tol=1e-6, verbose=False)
            Eu_c = rel_l2_error_lumped(m_c, uT_fom, res_c["u"][-1])
            J_c = cost_function(m_c, res_c["u"], res_c["Y"], yD, dt)
            dJ_c = abs(J_c - J_fom)
            Psi_y_s, Psi_p_s, _, _, _ = build_separate_pod_basis_lumped(m_c, res_fom["Y"], res_fom["P"], d_snap=d, l=l, dt=dt, time_weight=True)
            res_s = pdas_rom_coupled_fem(gamma, T, dt, m_c, A_c, y0, yD, Psi_y_s, Psi_p_s,maxit=25, tol=1e-6, verbose=False)
            Eu_s = rel_l2_error_lumped(m_c, uT_fom, res_s["u"][-1])
            J_s = cost_function(m_c, res_s["u"], res_s["Y"], yD, dt)
            dJ_s = abs(J_s - J_fom)
            rows.append([str(d),str(l),to_sci(Eu_c, 3),to_sci(Eu_s, 3),to_sci(dJ_c, 6),to_sci(dJ_s, 6),to_sci(res_c["runtime"], 3),to_sci(res_s["runtime"], 3),])
    write_latex_table(AB_DIR / "table_coupled_vs_separate_low_snap.tex",caption=("Coupled POD vs separate POD with few snapshots ($d\\in\\{3,5\\}$): " "relative control error $E_u(T)$ and cost gap $|J_{\\mathrm{ROM}}-J_{\\mathrm{FOM}}|$."),label="tab:coupled_vs_separate_low_snap",headers=[r"$d$", r"$l$",r"$E_u(T)$ (coupled)", r"$E_u(T)$ (separate)",r"$|J_{\mathrm{ROM}}-J_{\mathrm{FOM}}|$ (coupled)",r"$|J_{\mathrm{ROM}}-J_{\mathrm{FOM}}|$ (separate)","Runtime [s] (coupled)", "Runtime [s] (separate)"],rows=rows,align="cc|cc|cc|cc")
    print("Wrote:", (TAB_DIR / "table_coupled_vs_separate_low_snap.tex").resolve())
    print("J_FOM =", J_fom)
if __name__ == "__main__":
    run()

