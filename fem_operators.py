# fem_operators.py
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from fem_mesh import StructuredTriMesh
def _triangle_area_and_grads(xy: np.ndarray):
    """
    For a triangle with vertices (x1,y1),(x2,y2),(x3,y3),
    return area and gradients of P1 basis functions:
      grad(phi_i) = [b_i, c_i] / (2A)
    where:
      b1 = y2 - y3, c1 = x3 - x2, etc.
    """
    x1, y1 = xy[0]
    x2, y2 = xy[1]
    x3, y3 = xy[2]
    detJ = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    A = 0.5 * abs(detJ)
    if A <= 0:
        raise RuntimeError("Degenerate triangle area")
    b = np.array([y2 - y3, y3 - y1, y1 - y2], dtype=float)
    c = np.array([x3 - x2, x1 - x3, x2 - x1], dtype=float)
    grads = np.stack([b, c], axis=1) / (2.0 * A)  # (3,2)
    return A, grads
def assemble_mass_lumped_and_stiffness_interior(mesh: StructuredTriMesh):
    """
    Assemble:
      - m_int: lumped mass vector on interior DOFs (size n_int)
      - A_int: stiffness matrix on interior DOFs (CSR), for Laplacian term:
          \int grad(phi_i)·grad(phi_j) dx
    Dirichlet boundary nodes are eliminated (interior-only system).

    Notes:
      - Mass is lumped: each triangle contributes A/3 to each of its 3 vertices.
      - Stiffness uses standard P1 local matrix:
          K_ij = A * grad(phi_i)·grad(phi_j)
    """
    n_int = mesh.n_int
    m_int = np.zeros(n_int, dtype=float)
    rows, cols, data = [], [], []
    for tri in mesh.tris:
        xy = mesh.nodes[tri]  # (3,2)
        area, grads = _triangle_area_and_grads(xy)
        Kel = np.zeros((3, 3), dtype=float)
        for i in range(3):
            for j in range(3):
                Kel[i, j] = area * float(np.dot(grads[i], grads[j]))
        mel = (area / 3.0) * np.ones(3, dtype=float)
        for a_local, a_global in enumerate(tri):
            ia = mesh.g2i[a_global]
            if ia >= 0:
                m_int[ia] += mel[a_local]
        for a_local, a_global in enumerate(tri):
            ia = mesh.g2i[a_global]
            if ia < 0:
                continue
            for b_local, b_global in enumerate(tri):
                ib = mesh.g2i[b_global]
                if ib < 0:
                    continue
                rows.append(ia)
                cols.append(ib)
                data.append(Kel[a_local, b_local])

    A_int = sp.coo_matrix((data, (rows, cols)), shape=(n_int, n_int)).tocsr()
    A_int.sum_duplicates()
    return m_int, A_int
