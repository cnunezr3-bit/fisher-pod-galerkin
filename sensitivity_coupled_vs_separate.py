from config import MU, T_FINAL, DT, NX_COARSE, D_SNAP, L_LIST, TAB_DIR
from fem_mesh import build_structured_tri_mesh
from fem_operators import assemble_mass_lumped_and_stiffness_interior
from fom import pdas_fom_fem
from pod import build_coupled_pod_basis_lumped
from pod_separate import build_separate_pod_basis_lumped
from rom import pdas_rom_coupled_fem
from io_utils import write_latex_table, to_sci
import numpy as np




def run_comparison():

    gamma = MU
    T = T_FINAL
    dt = DT

    mesh = build_structured_tri_mesh(NX_COARSE)
    m_c, A_c = assemble_mass_lumped_and_stiffness_interior(mesh)

    y0 = np.zeros(mesh.n_int)
    for idx_int, g in enumerate(mesh.interior):
        x, y = mesh.nodes[g]
        y0[idx_int] = 16.0 * x * y * (x - 1.0) * (y - 1.0)

    yD = 0.5 * np.ones(mesh.n_int)
    res_fom = pdas_fom_fem(gamma, T, dt, m_c, A_c, y0, yD)

    rows = []

    for l in L_LIST:
        Psi_y_c, Psi_p_c, _, _, _ = build_coupled_pod_basis_lumped(
            m_c, res_fom["Y"], res_fom["P"],
            d_snap=D_SNAP, l=l, dt=dt
        )

        res_rom_c = pdas_rom_coupled_fem(gamma, T, dt, m_c, A_c,y0, yD,Psi_y_c, Psi_p_c)
        Psi_y_s, Psi_p_s, _, _, _ = build_separate_pod_basis_lumped(m_c, res_fom["Y"], res_fom["P"],d_snap=D_SNAP, l=l, dt=dt)
        res_rom_s = pdas_rom_coupled_fem(gamma, T, dt, m_c, A_c,y0, yD,Psi_y_s, Psi_p_s)
        # errors
        y_err_c = np.linalg.norm(res_fom["Y"][-1] - res_rom_c["Y"][-1])
        y_err_s = np.linalg.norm(res_fom["Y"][-1] - res_rom_s["Y"][-1])
        rows.append([str(l),to_sci(y_err_c,3),to_sci(y_err_s,3),to_sci(res_rom_c["runtime"],3),to_sci(res_rom_s["runtime"],3)])
    write_latex_table(TAB_DIR / "table_coupled_vs_separate.tex",caption="Coupled POD vs Separate POD comparison.",label="tab:coupled_vs_separate",headers=["$l$","Coupled error","Separate error","Coupled runtime","Separate runtime"],rows=rows)


if __name__ == "__main__":
    run_comparison()

