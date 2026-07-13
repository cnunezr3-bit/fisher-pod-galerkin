# main.py
from __future__ import annotations
import numpy as np
from config import (CMAP, SAVE_EPS, SHOW_FIGS, FIG_DIR, TAB_DIR,MU, T_FINAL, DT,NX_COARSE, NX_FINE,D_SNAP, L_LIST, PLOT_POD_MODES, PLOT_WITH_BOUNDARY)
from fem_mesh import build_structured_tri_mesh
from fem_operators import assemble_mass_lumped_and_stiffness_interior
from fom import pdas_fom_fem, posteriori_indicator_xi
from pod import build_coupled_pod_basis_lumped
from rom import pdas_rom_coupled_fem
from io_utils import to_sci, write_latex_table
from plotting import (plot_field_2d, plot_field_3d, plot_pod_modes_3d )
def l2_norm_lumped(m_int: np.ndarray, v: np.ndarray) -> float:
    return float(np.sqrt(np.sum(m_int * (v * v))))
def l2_error_lumped(m_int: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> float:
    return l2_norm_lumped(m_int, v1 - v2)
def rel_l2_error_lumped(m_int: np.ndarray, v_ref: np.ndarray, v_app: np.ndarray) -> float:
    return l2_error_lumped(m_int, v_ref, v_app) / (l2_norm_lumped(m_int, v_ref) + 1e-14)
def build_y0_on_interior(mesh):
    """
    y0(x,y) = 16 x y (x-1)(y-1) on interior nodes.
    """
    Nx = mesh.Nx
    n = mesh.n_int
    y0 = np.zeros(n, dtype=float)
    for idx_int, g in enumerate(mesh.interior):
        x, y = mesh.nodes[g]
        y0[idx_int] = 16.0 * x * y * (x - 1.0) * (y - 1.0)
    return y0
def main():
    mu = MU
    T = T_FINAL
    dt = DT
    mesh_c = build_structured_tri_mesh(NX_COARSE)
    m_c, A_c = assemble_mass_lumped_and_stiffness_interior(mesh_c)
    mesh_f = build_structured_tri_mesh(NX_FINE)
    m_f, A_f = assemble_mass_lumped_and_stiffness_interior(mesh_f)
    y0c = build_y0_on_interior(mesh_c)
    y0f = build_y0_on_interior(mesh_f)
    yDc = 0.5 * np.ones(mesh_c.n_int, dtype=float)
    yDf = 0.5 * np.ones(mesh_f.n_int, dtype=float)
    print("\n=== FOM(FEM) (coarse) ===")
    res_c = pdas_fom_fem(mu, T, dt, m_c, A_c, y0c, yDc, maxit=25, tol=1e-8, verbose=True)
    print(f"Coarse FOM runtime: {res_c['runtime']:.3e} s")
    print(f"Coarse FOM PDAS iters: {res_c['pdas_iters']}")
    print(f"Coarse FOM avg Newton/step: {res_c['avg_newton_per_step']:.2f}")
    print("\n=== FOM(FEM) (fine) ===")
    res_f = pdas_fom_fem(mu, T, dt, m_f, A_f, y0f, yDf, maxit=25, tol=1e-8, verbose=True)
    print(f"Fine FOM runtime: {res_f['runtime']:.3e} s")
    print(f"Fine FOM PDAS iters: {res_f['pdas_iters']}")
    print(f"Fine FOM avg Newton/step: {res_f['avg_newton_per_step']:.2f}")
    for grid_tag, Nx, res in [("coarse", NX_COARSE, res_c), ("fine", NX_FINE, res_f)]:
        uT, yT, pT = res["u"][-1], res["Y"][-1], res["P"][-1]
        plot_field_2d(uT, Nx, "u", f"FOM_{grid_tag}_uT_2D_Nx{Nx}",
                      cmap=CMAP, fig_dir=FIG_DIR, save_eps=SAVE_EPS, with_boundary=PLOT_WITH_BOUNDARY)
        plot_field_2d(yT, Nx, "y", f"FOM_{grid_tag}_yT_2D_Nx{Nx}",
                      cmap=CMAP, fig_dir=FIG_DIR, save_eps=SAVE_EPS, with_boundary=PLOT_WITH_BOUNDARY)
        plot_field_2d(pT, Nx, "p", f"FOM_{grid_tag}_pT_2D_Nx{Nx}",
                      cmap=CMAP, fig_dir=FIG_DIR, save_eps=SAVE_EPS, with_boundary=PLOT_WITH_BOUNDARY)

        plot_field_3d(uT, Nx, "u", f"FOM_{grid_tag}_uT_3D_Nx{Nx}",
                      cmap=CMAP, fig_dir=FIG_DIR, save_eps=SAVE_EPS, with_boundary=PLOT_WITH_BOUNDARY)
        plot_field_3d(yT, Nx, "y", f"FOM_{grid_tag}_yT_3D_Nx{Nx}",
                      cmap=CMAP, fig_dir=FIG_DIR, save_eps=SAVE_EPS, with_boundary=PLOT_WITH_BOUNDARY)
        plot_field_3d(pT, Nx, "p", f"FOM_{grid_tag}_pT_3D_Nx{Nx}",
                      cmap=CMAP, fig_dir=FIG_DIR, save_eps=SAVE_EPS, with_boundary=PLOT_WITH_BOUNDARY)
    print("\n=== Coupled POD (built from coarse FOM(FEM) snapshots) ===")
    l_max = max(L_LIST + [PLOT_POD_MODES])
    Psi_y_max, Psi_p_max, Psi2_max, evals, snap_idx = build_coupled_pod_basis_lumped(m_c, res_c["Y"], res_c["P"],d_snap=D_SNAP, l=l_max, dt=dt, time_weight=True)
    print(f"Snapshots requested: {D_SNAP}")
    print(f"Snapshots used: {len(snap_idx)}")
    print(f"Top-5 eigenvalues: {evals[:5]}")
    rows_eigs = []
    for i, lam in enumerate(evals[:max(30, l_max)]):
        rows_eigs.append([str(i + 1), to_sci(float(lam), 6)])
    write_latex_table(TAB_DIR / "table_pod_eigenvalues.tex",caption="Coupled POD eigenvalues (descending) computed from coarse FEM snapshots (lumped mass inner product).",label="tab:pod_eigs",headers=["index", r"$\lambda_i$"],rows=rows_eigs,align="cc")plot_pod_modes_3d(Nx=NX_COARSE, Psi_y=Psi_y_max, Psi_p=Psi_p_max, num_modes=PLOT_POD_MODES,cmap=CMAP, fig_dir=FIG_DIR, save_eps=SAVE_EPS,with_boundary=True,prefix=f"POD_FEM_Nx{NX_COARSE}_dsnap{D_SNAP}")
    n_c = mesh_c.n_int
    n_f = mesh_f.n_int
    rows_fom = [["FOM coarse (FEM)", str(NX_COARSE), str(n_c), str(res_c["Nt"]), to_sci(res_c["runtime"], 3),str(res_c["pdas_iters"]), f"{res_c['avg_newton_per_step']:.2f}"],["FOM fine (FEM)", str(NX_FINE), str(n_f), str(res_f["Nt"]), to_sci(res_f["runtime"], 3),str(res_f["pdas_iters"]), f"{res_f['avg_newton_per_step']:.2f}"],]
    rows_rom = []
    rows_err = []
    for l in L_LIST:
        Psi_y, Psi_p, _, evals_all, snap_idx2 = build_coupled_pod_basis_lumped(m_c, res_c["Y"], res_c["P"],d_snap=D_SNAP, l=l, dt=dt, time_weight=True)
        energy = float(np.sum(evals_all[:l]) / (np.sum(evals_all) + 1e-30))
        cond_y = float(np.linalg.cond(Psi_y.T @ Psi_y))
        cond_p = float(np.linalg.cond(Psi_p.T @ Psi_p))
        print(f"\n=== ROM(FEM) (coarse) l={l} ===")
        res_rom = pdas_rom_coupled_fem(mu, T, dt, m_c, A_c, y0c, yDc, Psi_y, Psi_p,
                                       maxit=25, tol=1e-6, verbose=True)
        y_err = l2_error_lumped(m_c, res_c["Y"][-1], res_rom["Y"][-1])
        p_err = l2_error_lumped(m_c, res_c["P"][-1], res_rom["P"][-1])
        u_err_rel = rel_l2_error_lumped(m_c, res_c["u"][-1], res_rom["u"][-1])
        xi = posteriori_indicator_xi(res_rom["u"], res_rom["Y"], res_rom["P"])
        xi_T = l2_norm_lumped(m_c, xi[-1])
        uT, yT, pT = res_rom["u"][-1], res_rom["Y"][-1], res_rom["P"][-1]
        tag = f"ROM_FEM_coarse_l{l:02d}_Nx{NX_COARSE}_dsnap{D_SNAP}"
        plot_field_2d(uT, NX_COARSE, "u", f"{tag}_uT_2D",
                      cmap=CMAP, fig_dir=FIG_DIR, save_eps=SAVE_EPS, with_boundary=PLOT_WITH_BOUNDARY)
        plot_field_2d(yT, NX_COARSE, "y", f"{tag}_yT_2D",
                      cmap=CMAP, fig_dir=FIG_DIR, save_eps=SAVE_EPS, with_boundary=PLOT_WITH_BOUNDARY)
        plot_field_2d(pT, NX_COARSE, "p", f"{tag}_pT_2D",
                      cmap=CMAP, fig_dir=FIG_DIR, save_eps=SAVE_EPS, with_boundary=PLOT_WITH_BOUNDARY)

        plot_field_3d(uT, NX_COARSE, "u", f"{tag}_uT_3D",
                      cmap=CMAP, fig_dir=FIG_DIR, save_eps=SAVE_EPS, with_boundary=PLOT_WITH_BOUNDARY)
        plot_field_3d(yT, NX_COARSE, "y", f"{tag}_yT_3D",
                      cmap=CMAP, fig_dir=FIG_DIR, save_eps=SAVE_EPS, with_boundary=PLOT_WITH_BOUNDARY)
        plot_field_3d(pT, NX_COARSE, "p", f"{tag}_pT_3D",
                      cmap=CMAP, fig_dir=FIG_DIR, save_eps=SAVE_EPS, with_boundary=PLOT_WITH_BOUNDARY)
        rows_rom.append([
            str(l),
            str(n_c),
            str(2 * l),
            str(len(snap_idx2)),
            f"{energy:.6f}",
            to_sci(res_rom["runtime"], 3),
            str(res_rom["pdas_iters"]),
            f"{res_rom['avg_newton_per_step']:.2f}",
            to_sci(cond_y, 2),
            to_sci(cond_p, 2),])
        rows_err.append([
            str(l),
            to_sci(y_err, 3),
            to_sci(p_err, 3),
            to_sci(u_err_rel, 3),
            to_sci(xi_T, 3),])
    write_latex_table(
        TAB_DIR / "table_fom_summary.tex",
        caption="FOM(FEM) performance summary on coarse and fine grids (P1 FEM with lumped mass).",
        label="tab:fom_summary",
        headers=["Model", "$N_x$", "DOF $n$", "$N_t$", "Runtime [s]", "PDAS iters", "Avg Newton/step"],
        rows=rows_fom,align="lcccccc")
    write_latex_table(
        TAB_DIR / "table_rom_summary.tex",
        caption="ROM(FEM) performance summary on the coarse grid for different reduced dimensions $l$ (coupled POD, lumped mass inner product).",
        label="tab:rom_summary",
        headers=["$l$", "FOM DOF $n$", "ROM DOF $2l$", "snapshots", "energy",
                 "Runtime [s]", "PDAS iters", "Avg Newton/step", r"$\kappa(P_y^TP_y)$", r"$\kappa(P_p^TP_p)$"],rows=rows_rom,align="cccccccccc")

    write_latex_table(
        TAB_DIR / "table_errors.tex",
        caption="Errors at final time $T$ for ROM(FEM) vs coarse FOM(FEM) (lumped mass $L^2$).",
        label="tab:rom_errors",
        headers=["$l$",r"$\|y_{\mathrm{FOM}}(T)-y_{\mathrm{ROM}}(T)\|_{L^2}$",
                 r"$\|p_{\mathrm{FOM}}(T)-p_{\mathrm{ROM}}(T)\|_{L^2}$",
                 r"$E_u(T)$ (rel.)",
                 r"$\|\xi(T)\|_{L^2}$"],rows=rows_err,align="c|cccc")
    print("\n--- Outputs written ---")
    print(f"Figures (EPS):   {FIG_DIR.resolve()}")
    print(f"LaTeX tables:    {TAB_DIR.resolve()}")
    print("  - table_pod_eigenvalues.tex")
    print("  - table_fom_summary.tex")
    print("  - table_rom_summary.tex")
    print("  - table_errors.tex")
    if SHOW_FIGS:
        import matplotlib.pyplot as plt
        plt.show()
if __name__ == "__main__":
    main()
