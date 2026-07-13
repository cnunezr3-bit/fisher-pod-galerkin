# plotting.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def save_fig(fig, fig_dir: Path, filename_base: str, save_eps: bool):
    if not save_eps:
        return
    out = fig_dir / f"{filename_base}.eps"
    fig.savefig(out, format="eps", bbox_inches="tight")
    plt.close(fig)

def grid_from_vec_interior(v: np.ndarray, Nx: int):
    n = Nx - 2
    return v.reshape((n, n))

def grid_with_boundary_from_vec(v: np.ndarray, Nx: int, bc_value=0.0):
    n = Nx - 2
    Ufull = np.full((Nx, Nx), bc_value, dtype=float)
    Ufull[1:-1, 1:-1] = v.reshape((n, n))
    return Ufull

def full_grid_coords(Nx: int):
    x = np.linspace(0.0, 1.0, Nx)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return X, Y

def _coords_for_plot(Nx: int, with_boundary: bool):
    if with_boundary:
        return full_grid_coords(Nx)
    n = Nx - 2
    x = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    return X, Y

def _field_to_plot_grid(v_interior: np.ndarray, Nx: int, with_boundary: bool):
    if with_boundary:
        return grid_with_boundary_from_vec(v_interior, Nx, bc_value=0.0)
    return grid_from_vec_interior(v_interior, Nx)

def plot_field_2d(v_interior, Nx, kind, filename_base, *,
                  cmap: str, fig_dir: Path, save_eps: bool, with_boundary: bool):
    G = _field_to_plot_grid(v_interior, Nx, with_boundary)
    fig, ax = plt.subplots(1, 1, figsize=(4.2, 3.6))

    if kind == "u":
        vmin, vmax = 0.0, 1.0
    elif kind == "y":
        vmin, vmax = 0.0, max(1.0, float(np.max(G)))
    elif kind == "p":
        pmax = float(np.max(np.abs(G))) + 1e-30
        vmin, vmax = -pmax, pmax
    else:
        vmin, vmax = None, None

    im = ax.imshow(G, origin="lower", extent=(0, 1, 0, 1), aspect="auto",
                   cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("")  # no title
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_fig(fig, fig_dir, filename_base, save_eps)

def plot_field_3d(v_interior, Nx, kind, filename_base, *,
                  cmap: str, fig_dir: Path, save_eps: bool, with_boundary: bool):
    X, Y = _coords_for_plot(Nx, with_boundary)
    G = _field_to_plot_grid(v_interior, Nx, with_boundary)

    if kind == "u":
        vmin, vmax = 0.0, 1.0
    elif kind == "y":
        vmin, vmax = 0.0, max(1.0, float(np.max(G)))
    elif kind == "p":
        pmax = float(np.max(np.abs(G))) + 1e-30
        vmin, vmax = -pmax, pmax
    else:
        vmin, vmax = None, None

    fig = plt.figure(figsize=(4.8, 4.0))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, G, cmap=cmap, linewidth=0, antialiased=True,
                           vmin=vmin, vmax=vmax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("")  # no title
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.08)
    fig.tight_layout()
    save_fig(fig, fig_dir, filename_base, save_eps)

def plot_pod_modes_3d(Nx, Psi_y, Psi_p, num_modes, *,
                      cmap: str, fig_dir: Path, save_eps: bool,
                      with_boundary: bool, prefix: str):
    m = min(num_modes, Psi_y.shape[1], Psi_p.shape[1])
    for i in range(m):
        plot_field_3d(Psi_y[:, i], Nx, "p",
                      f"{prefix}_PsiY_mode{i+1:02d}_3D",
                      cmap=cmap, fig_dir=fig_dir, save_eps=save_eps,
                      with_boundary=with_boundary)
        plot_field_3d(Psi_p[:, i], Nx, "p",
                      f"{prefix}_PsiP_mode{i+1:02d}_3D",
                      cmap=cmap, fig_dir=fig_dir, save_eps=save_eps,
                      with_boundary=with_boundary)

