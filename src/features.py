from pathlib import Path
import numpy as np

FT_CACHE = Path("data/features")

FEATURE_NAMES = (
    ["c1_r", "c1_g", "c1_b", "c2_h", "c2_s", "c2_v"] +
    [f"c3_h{i}" for i in range(5)] + ["c3_ent"] +
    [f"c4_s{i}" for i in range(3)] + ["c4_ent"] +
    ["l1_x", "l1_y", "l2_x10", "l2_x90", "l2_y10", "l2_y90",
     "l3_yh10", "l3_yh90", "l4_nsp", "l5_hull", "l6_compact", "l7_contig"] +
    [f"t1_d{i}" for i in range(12)] + ["t2_mean", "t3_argmax", "t4_contrast"] +
    ["g1_lines", "g2_parallel"] +
    [f"g3_h{i}" for i in range(12)] + ["g3_ent"] +
    ["g4_right", "g5_above"] +
    [f"g6_{i}" for i in range(8)] + [f"g7_{i}" for i in range(8)] +
    ["g8_gx", "g8_gy"]
)

GROUPS = {"color": (0,16), "texture": (16,31), "location": (31,43), "geometry": (43,78)}
