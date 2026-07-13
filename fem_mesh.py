# fem_mesh.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class StructuredTriMesh:
    """
    Structured mesh on (0,1)^2 with Nx nodes per direction (including boundary),
    each square split into two triangles.
    Nodes are ordered lexicographically: (i,j) -> idx = i*Nx + j, with
    i = 0..Nx-1 (x-direction), j = 0..Nx-1 (y-direction).
    Interior DOFs are nodes with i=1..Nx-2, j=1..Nx-2.
    We store an interior mapping consistent with a (Nx-2)x(Nx-2) grid:
        interior_index(i,j) = (i-1)*(Nx-2) + (j-1)
    """
    Nx: int
    nodes: np.ndarray      # (Nnodes, 2)
    tris: np.ndarray       # (Ntri, 3) node indices
    interior: np.ndarray   # (n_int,) global node indices
    g2i: np.ndarray        # (Nnodes,) map global->interior index, -1 if boundary
    n_int: int
    h: float
def build_structured_tri_mesh(Nx: int) -> StructuredTriMesh:
    if Nx < 3:
        raise ValueError("Nx must be >= 3")
    x = np.linspace(0.0, 1.0, Nx)
    h = x[1] - x[0]
    nodes = np.zeros((Nx * Nx, 2), dtype=float)
    for i in range(Nx):
        for j in range(Nx):
            idx = i * Nx + j
            nodes[idx, 0] = x[i]
            nodes[idx, 1] = x[j]
    tris = []
    for i in range(Nx - 1):
        for j in range(Nx - 1):
            n00 = i * Nx + j
            n10 = (i + 1) * Nx + j
            n01 = i * Nx + (j + 1)
            n11 = (i + 1) * Nx + (j + 1)
            tris.append([n00, n10, n11])
            tris.append([n00, n11, n01])
    tris = np.array(tris, dtype=int)
    interior = []
    g2i = -np.ones(Nx * Nx, dtype=int)
    cnt = 0
    for i in range(1, Nx - 1):
        for j in range(1, Nx - 1):
            g = i * Nx + j
            interior.append(g)
            g2i[g] = cnt
            cnt += 1
    interior = np.array(interior, dtype=int)
    return StructuredTriMesh(
        Nx=Nx,
        nodes=nodes,
        tris=tris,
        interior=interior,
        g2i=g2i,
        n_int=(Nx - 2) * (Nx - 2),
        h=h,
    )

