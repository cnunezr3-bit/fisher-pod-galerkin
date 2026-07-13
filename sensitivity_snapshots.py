from __future__ import annotations
import numpy as np
from config import (T_FINAL, DT, MU,NX_COARSE,L_LIST,TAB_DIR)
from fem_mesh import build_structured_tri_mesh
from fem_operators import assemble_mass_lumped_and_stiffness_interior
from fom import pdas_fom_fem, posteriori_indicator_xi, cost_function
from pod import build_coupled_pod_basis_lumped
from rom import pdas_rom_coupled_fem
from io_utils import to_sci, write_latex_table


def l2_norm_lumped(m_int, v):
    return float(np.sqrt(np.sum(m_int * (v * v))))

def l2_error_lumped(m_int, v1, v2):
    return l2_norm_lumped(m_int, v1 - v2)

def rel_l2_error_lumped(m_int, v_ref, v_app):
    return l2_error_lumped(m_int, v_ref, v_app) / (l2_norm_lumped(m_int, v_ref) + 1e-14)

def build_y0_on_interior(mesh):
    y0 = np.zeros(mesh.n_int)
    for idx_int, g in enumerate(mesh.interior):
        x, y = mesh.nodes[g]
        y0[idx_int] = 16.0 * x * y * (x - 1.0) * (y - 1.0)
    return y0
def run_sensitivity_snapshots():
    DSNAP_LIST = [3, 5, 10, 20, 40]
    gamma = MU
    T = T_FINAL
    dt = DT
    mesh = build_structured_tri_mesh(NX_COARSE)
    m_c, A_c = assemble_mass_lumped_and_stiffness_interior(mesh)
    y0 = build_y0_on_interior(mesh)
    yD = 0.5 * np.ones(mesh.n_int)
    res_fom = pdas_fom_fem(gamma, T, dt, m_c, A_c, y0, yD,maxit=25, tol=1e-8, verbose=True)
    rows_energy = []
    rows_errors = []
    for d_snap in DSNAP_LIST:
        print(f"\n==========================")
        print(f" Snapshot sensitivity: d = {d_snap}")
        print(f"==========================\n")
        l_max = max(L_LIST)
        Psi_y_max, Psi_p_max, _, evals, snap_idx = \
            build_coupled_pod_basis_lumped(
                m_c, res_fom["Y"], res_fom["P"],
                d_snap=d_snap, l=l_max,dt=dt, time_weight=True)
        total_e = float(np.sum(evals) + 1e-30)
        for l in L_LIST:
            energy = float(np.sum(evals[:l]) / total_e)
            rows_energy.append([str(d_snap), str(l),f"{energy:.6f}",str(len(snap_idx))])

        for l in L_LIST:
            Psi_y = Psi_y_max[:, :l]
            Psi_p = Psi_p_max[:, :l]
            res_rom = pdas_rom_coupled_fem(gamma, T, dt, m_c, A_c,y0, yD,Psi_y, Psi_p,maxit=25, tol=1e-6, verbose=True)
            y_err = l2_error_lumped(m_c, res_fom["Y"][-1], res_rom["Y"][-1])
            p_err = l2_error_lumped(m_c, res_fom["P"][-1], res_rom["P"][-1])
            u_err = rel_l2_error_lumped(m_c, res_fom["u"][-1], res_rom["u"][-1])
            xi = posteriori_indicator_xi(res_rom["u"], res_rom["Y"], res_rom["P"])
            xi_T = l2_norm_lumped(m_c, xi[-1])

            rows_errors.append([str(d_snap),
                str(l),
                to_sci(y_err, 3),
                to_sci(p_err, 3),
                to_sci(u_err, 3),
                to_sci(xi_T, 3),
                str(res_rom["pdas_iters"]),
                to_sci(res_rom["runtime"], 3)])

    write_latex_table(
        TAB_DIR / "table_sens_snap_energy.tex",
        caption="Sensitivity to the number of snapshots $d$: POD cumulative energy.",
        label="tab:sens_snap_energy",
        headers=[r"$d$", r"$l$", "energy", "snapshots"],
        rows=rows_energy,
        align="cccc")

    write_latex_table(
        TAB_DIR / "table_sens_snap_errors.tex",
        caption="Sensitivity to the number of snapshots $d$: ROM errors at final time $T$.",
        label="tab:sens_snap_errors",
        headers=[r"$d$", r"$l$",
                 r"$\|y_{\mathrm{FOM}}-y_{\mathrm{ROM}}\|_{L^2}$",
                 r"$\|p_{\mathrm{FOM}}-p_{\mathrm{ROM}}\|_{L^2}$",
                 r"$E_u$ (rel.)",
                 r"$\|\xi(T)\|_{L^2}$",
                 "PDAS iters",
                 "Runtime [s]"],
        rows=rows_errors,
        align="cc|ccccc|cc")


if __name__ == "__main__":
    run_sensitivity_snapshots()

