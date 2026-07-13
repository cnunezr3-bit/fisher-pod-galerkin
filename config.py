# config.py
from pathlib import Path
# ===============
# USER SETTINGS
# ===============
CMAP = "bone"          
SAVE_EPS = True
SHOW_FIGS = False
FIG_DIR = Path("graficos")
TAB_DIR = Path("tables")
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)
# PDE / discretization parameters
MU = 1e-3
T_FINAL = 0.9
DT = 3.0e-3
# FEM grid: number of nodes per direction (including boundary)
NX_COARSE = 17
NX_FINE = 41
# POD / ROM configuration
D_SNAP = 10
L_LIST = [1, 3, 5, 10, 15]
PLOT_POD_MODES = 6
PLOT_WITH_BOUNDARY = False

