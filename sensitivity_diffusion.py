from __future__ import annotations
import numpy as np
from config import (CMAP, SAVE_EPS, SHOW_FIGS, FIG_DIR, TAB_DIR,T_FINAL, DT,NX_COARSE,D_SNAP, L_LIST, PLOT_WITH_BOUNDARY)
from fem_mesh import build_structured_tri_mesh
from fem_operators import assemble_mass_lumped_and_stiffness_interior
from fom import pdas_fom_fem, posteriori_indicator_xi, cost_function
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
def run_sensitivity_diffusion():
    GAMMA_LIST = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    T = T_FINAL
    dt = DT
    mesh_c = build_structured_tri_mesh(NX_COARSE)
    m_c, A_c = assemble_mass_lumped_and_stiffness_interior(mesh_c)
    y0c = build_y0_on_interior(mesh_c)
    yDc = 0.5 * np.ones(mesh_c.n_int, dtype=float)
    rows = []
    rows_energy = []
    for gamma in GAMMA_LIST:
        print(f"\n===============================")
        print(f" Diffusion sensitivity: gamma={gamma:.2e}")
        print(f"===============================\n")
        res_fom = pdas_fom_fem(gamma, T, dt, m_c, A_c, y0c, yDc, maxit=25, tol=1e-8, verbose=True)
        l_max = max(L_LIST)
        Psi_y_max, Psi_p_max, _, evals, snap_idx = build_coupled_pod_basis_lumped(m_c, res_fom["Y"], res_fom["P"],d_snap=D_SNAP, l=l_max, dt=dt, time_weight=True)
        total_e = float(np.sum(evals) + 1e-30)
        for l in L_LIST:
            ecap = float(np.sum(evals[:l]) / total_e)
            rows_energy.append([to_sci(gamma, 2), str(l), f"{ecap:.6f}", str(len(snap_idx))])
        for l in L_LIST:
            Psi_y = Psi_y_max[:, :l]
            Psi_p = Psi_p_max[:, :l]
            res_rom = pdas_rom_coupled_fem(gamma, T, dt, m_c, A_c, y0c, yDc, Psi_y, Psi_p,maxit=25, tol=1e-6, verbose=True)
            y_err = l2_error_lumped(m_c, res_fom["Y"][-1], res_rom["Y"][-1])
            p_err = l2_error_lumped(m_c, res_fom["P"][-1], res_rom["P"][-1])
            u_err_rel = rel_l2_error_lumped(m_c, res_fom["u"][-1], res_rom["u"][-1])
            xi = posteriori_indicator_xi(res_rom["u"], res_rom["Y"], res_rom["P"])
            xi_T = l2_norm_lumped(m_c, xi[-1])
            J_rom = cost_function(m_c, res_rom["u"], res_rom["Y"], yDc, dt)
            rows.append([to_sci(gamma, 2),str(l),to_sci(y_err, 3),to_sci(p_err, 3),to_sci(u_err_rel, 3),to_sci(xi_T, 3),to_sci(J_rom, 6),
                str(res_rom["pdas_iters"]),to_sci(res_rom["runtime"], 3),])
    write_latex_table(TAB_DIR / "table_sens_diffusion_errors.tex",caption="Sensitivity to diffusion parameter $\\gamma$: ROM errors at final time $T$ vs coarse FOM reference (lumped $L^2$).",
        label="tab:sens_diffusion_errors",headers=[r"$\gamma$", r"$l$",r"$\|y_{\mathrm{FOM}}(T)-y_{\mathrm{ROM}}(T)\|_{L^2}$", r"$\|p_{\mathrm{FOM}}(T)-p_{\mathrm{ROM}}(T)\|_{L^2}$",r"$E_u(T)$ (rel.)",r"$\|\xi(T)\|_{L^2}$",r"$J_{\mathrm{ROM}}$","PDAS iters","Runtime [s]"],rows=rows,align="cc|ccccc|ccc")
    write_latex_table(TAB_DIR / "table_sens_diffusion_energy.tex",caption="Sensitivity to diffusion parameter $\\gamma$: POD cumulative energy for each reduced dimension $l$ (coupled POD).",
        label="tab:sens_diffusion_energy",headers=[r"$\gamma$", r"$l$", "energy", "snapshots"],rows=rows_energy,align="cccc")
    print("\n--- Sensitivity diffusion outputs written ---")
    print(f"LaTeX tables: {TAB_DIR.resolve()}")
    print("  - table_sens_diffusion_errors.tex")
    print("  - table_sens_diffusion_energy.tex")

if __name__ == "__main__":
    run_sensitivity_diffusion()

