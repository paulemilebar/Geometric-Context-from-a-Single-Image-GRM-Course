"""
03b_segmentation_hypotheses.py  —  Hoiem 2005, Section 3.1  (étape manquante)
══════════════════════════════════════════════════════════════════════════════

Pipeline implémenté :

  Étape A — Modèle d'affinité  (50 images d'entraînement)
  ─────────────────────────────────────────────────────────
  • Échantillonne 2 500 paires de superpixels de MÊME classe  (y=1)
             et  2 500 paires de superpixels de CLASSE DIFFÉRENTE (y=0)
  • Features d'entrée : |x_i − x_j|  (différence absolue des 78 features)
  • Modèle : AdaBoost logistique (SAMME.R) avec arbres de décision
  • Sortie  : log-odds  log P(même classe) / P(classe différente)

  Étape B — Algorithme glouton d'hypothèses  (300 images)
  ─────────────────────────────────────────────────────────
  Pour chaque nr ∈ {3,4,5,7,9,11,15,20,25}  (nombre de régions voulu) :
    1. Ordonner aléatoirement les superpixels
    2. Assigner les nr premiers à des régions distinctes
    3. Assigner chaque superpixel restant à la région maximisant
       la log-vraisemblance d'affinité moyenne avec ses membres
    4. Répéter l'étape 3  N_GREEDY_IT fois

  Étape C — Features de régions
  ──────────────────────────────
  Pour chaque région (union de masques superpixels) : calculer les 78 features
  sur le masque pixel de la région entière  (et non plus par superpixel).

Sorties :
  data/models/affinity_adaboost.pkl   — modèle d'affinité
  data/hypotheses/{stem}.npy          — hypothèses + features de régions

À lancer APRÈS 02_superpixels.py et 03_features.py.
Puis lancer 04b_multiH_classify.py pour l'évaluation complète Hoiem.
"""

import sys, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
import cv2
from scipy.spatial import ConvexHull
from scipy.stats import entropy

sys.path.insert(0, "src")
from dataset import HoiemDataset, LABEL_IDS
from superpixels import load_sp, load_ft

DATASET_DIR  = Path("dataset")
HYPO_CACHE   = Path("data/hypotheses"); HYPO_CACHE.mkdir(parents=True, exist_ok=True)
MODEL_DIR    = Path("data/models");     MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR      = Path("outputs");         OUT_DIR.mkdir(exist_ok=True)

# ── Hyper-paramètres Hoiem 2005 ───────────────────────────────────────────────
NR_VALUES     = [3, 4, 5, 7, 9, 11, 15, 20, 25]   # nr ∈ section 3.1
N_PAIRS_EACH  = 2_500   # paires same-label ET diff-label (section 3.1)
N_GREEDY_IT   = 3       # répétitions de l'étape 3 du glouton
HOMOG_TOL     = 0.05    # ≤ 5 % de pixels non-majoritaires = "homogène" (footnote 3)
AFF_N_EST     = 100     # estimateurs AdaBoost pour le modèle d'affinité
AFF_DEPTH     = 3       # max_depth ≈ 8 feuilles (cohérent avec section 3.2)


# ══════════════════════════════════════════════════════════════════════════════
# FeatureExtractor  (identique à 03_features.py — copié pour autonomie)
# ══════════════════════════════════════════════════════════════════════════════

HORIZON_RATIO = 0.40
N_DOOG        = 12
DOOG_ANGLES   = np.linspace(0, 180, N_DOOG, endpoint=False)


def _gauss2d(sigma, size):
    ax = np.arange(-size//2+1, size//2+1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return k / k.sum()


def _build_doog_filters(sigma1=1.0, sigma2=3.0, size=15):
    base = _gauss2d(sigma1, size) - _gauss2d(sigma2, size)
    out  = []
    for deg in DOOG_ANGLES:
        M = cv2.getRotationMatrix2D((size//2, size//2), deg, 1.0)
        out.append(cv2.warpAffine(base.astype(np.float32), M, (size, size)))
    return out


DOOG_FILTERS = _build_doog_filters()


def _apply_doog(gray):
    gf  = gray.astype(np.float32) / 255.0
    out = np.zeros((*gray.shape, N_DOOG), dtype=np.float32)
    for i, f in enumerate(DOOG_FILTERS):
        out[..., i] = np.abs(cv2.filter2D(gf, -1, f))
    return out


class FeatureExtractor:
    """Calcule les 78 features Hoiem sur un masque pixel quelconque."""

    def __init__(self, image):
        self.H, self.W = image.shape[:2]
        self.rgb  = image.astype(np.float32) / 255.0
        hsv       = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        self.hsv  = hsv / np.array([180., 255., 255.])
        self.gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.doog = _apply_doog(self.gray)
        self.horizon_y = self.H * HORIZON_RATIO
        self._lines_cache = None   # Hough lines calculées une seule fois

    # ── couleur ───────────────────────────────────────────────────────────────
    def _color(self, mask):
        f  = []
        f += self.rgb[mask].mean(axis=0).tolist()                      # C1 (3)
        f += self.hsv[mask].mean(axis=0).tolist()                      # C2 (3)
        h  = self.hsv[mask, 0]
        hh, _ = np.histogram(h, bins=5, range=(0, 1), density=True)
        hh /= hh.sum() + 1e-8
        f += hh.tolist() + [float(entropy(hh + 1e-8))]                # C3 (6)
        s  = self.hsv[mask, 1]
        sh, _ = np.histogram(s, bins=3, range=(0, 1), density=True)
        sh /= sh.sum() + 1e-8
        f += sh.tolist() + [float(entropy(sh + 1e-8))]                # C4 (4)
        return np.array(f, dtype=np.float32)                           # → 16

    # ── texture ───────────────────────────────────────────────────────────────
    def _texture(self, mask):
        t1 = self.doog[mask].mean(axis=0)                              # (12,)
        return np.array([*t1, t1.mean(), t1.argmax() / 11.0,
                         t1.max() - np.median(t1)], dtype=np.float32) # → 15

    # ── localisation et forme ─────────────────────────────────────────────────
    def _location(self, mask):
        ys, xs = np.where(mask)
        xn, yn = xs / self.W, ys / self.H
        hn     = self.horizon_y / self.H
        hull_sides, compact = 4.0, 0.5
        if len(ys) >= 4:
            try:
                hull       = ConvexHull(np.column_stack([xs, ys]).astype(np.float64))
                hull_sides = float(len(hull.vertices))
                compact    = float(np.clip(mask.sum() / (hull.volume + 1e-8), 0, 1))
            except Exception:
                pass
        n_sp_norm = float(len(ys)) / (self.H * self.W)
        x_range   = float(xs.max() - xs.min()) / self.W if len(xs) > 1 else 0.0
        return np.array([xn.mean(), yn.mean(),
                         np.percentile(xn, 10), np.percentile(xn, 90),
                         np.percentile(yn, 10), np.percentile(yn, 90),
                         np.percentile(yn, 10) - hn, np.percentile(yn, 90) - hn,
                         n_sp_norm, hull_sides, compact, x_range],
                        dtype=np.float32)                               # → 12

    # ── géométrie (lignes / points de fuite) ──────────────────────────────────
    def _get_lines(self):
        if self._lines_cache is not None:
            return self._lines_cache
        edges = cv2.Canny(self.gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50,
                                minLineLength=30, maxLineGap=10)
        self._lines_cache = (lines[:, 0, :] if lines is not None
                             else np.zeros((0, 4), np.float32))
        return self._lines_cache

    def _lines_in_mask(self, lines, mask):
        if not len(lines):
            return lines
        out = []
        for x1, y1, x2, y2 in lines:
            xc1 = int(np.clip(x1, 0, self.W-1))
            yc1 = int(np.clip(y1, 0, self.H-1))
            xc2 = int(np.clip(x2, 0, self.W-1))
            yc2 = int(np.clip(y2, 0, self.H-1))
            if mask[yc1, xc1] or mask[yc2, xc2]:
                out.append([x1, y1, x2, y2])
        return np.array(out, np.float32) if out else np.zeros((0, 4))

    @staticmethod
    def _angle(x1, y1, x2, y2):
        return np.arctan2(y2 - y1, x2 - x1) % np.pi

    def _parallel_frac(self, lines, tol=np.pi/8):
        if len(lines) < 2:
            return 0.0
        angs  = [self._angle(*l) for l in lines]
        total = pairs = 0
        for i in range(len(angs)):
            for j in range(i+1, len(angs)):
                d = abs(angs[i] - angs[j]); d = min(d, np.pi - d)
                total += 1
                if d < tol:
                    pairs += 1
        return pairs / total if total else 0.0

    @staticmethod
    def _intersect(l1, l2):
        x1, y1, x2, y2 = l1; x3, y3, x4, y4 = l2
        d = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(d) < 1e-8:
            return None
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / d
        return (x1 + t*(x2-x1), y1 + t*(y2-y1))

    def _intersections(self, lines):
        if len(lines) < 2:
            return np.zeros((0, 2))
        pts = [self._intersect(lines[i], lines[j])
               for i in range(len(lines)) for j in range(i+1, len(lines))]
        pts = [p for p in pts if p]
        return np.array(pts, np.float32) if pts else np.zeros((0, 2))

    def _far_frac(self, pts, cx, cy, thr):
        out = np.zeros(8, np.float32)
        if not len(pts):
            return out
        for k in range(8):
            a0    = k * np.pi / 4;  a1 = (k+1) * np.pi / 4
            angs  = np.arctan2(pts[:,1]-cy, pts[:,0]-cx) % (2*np.pi)
            dists = np.hypot(pts[:,0]-cx, pts[:,1]-cy)
            sect  = (angs >= a0) & (angs < a1)
            if sect.any():
                out[k] = float((dists[sect] > thr).mean())
        return out

    def _geometry(self, mask):
        lines  = self._get_lines()
        lm     = self._lines_in_mask(lines, mask)
        cx, cy = self.W / 2, self.H / 2
        diag   = np.hypot(self.W, self.H)

        g1  = float(len(lm)) / max(1, mask.sum() / 100)
        g2  = self._parallel_frac(lm)
        pts = self._intersections(lm)

        if len(pts):
            angs = np.degrees(np.arctan2(pts[:,1]-cy, pts[:,0]-cx)) % 360
            hist, _ = np.histogram(angs, bins=12, range=(0, 360))
            hn  = hist.astype(np.float32); hn /= hn.sum() + 1e-8
            g3  = np.append(hn, entropy(hn + 1e-8))
            g4  = float((pts[:,0] > cx).mean())
            g5  = float((pts[:,1] < cy).mean())
        else:
            g3  = np.zeros(13, np.float32)
            g4  = g5 = 0.5

        g6 = self._far_frac(pts, cx, cy, 1.5 * diag)
        g7 = self._far_frac(pts, cx, cy, 5.0 * diag)

        ys, xs = np.where(mask)
        if len(ys) >= 4:
            resp = self.doog[ys, xs].mean(axis=1)
            xn, yn = xs / self.W, ys / self.H
            gx = float(np.corrcoef(xn, resp)[0, 1]) if xn.std() > 1e-8 else 0.0
            gy = float(np.corrcoef(yn, resp)[0, 1]) if yn.std() > 1e-8 else 0.0
            gx = 0.0 if np.isnan(gx) else gx
            gy = 0.0 if np.isnan(gy) else gy
        else:
            gx = gy = 0.0

        return np.concatenate([[g1, g2], g3, [g4, g5], g6, g7, [gx, gy]]).astype(np.float32)  # → 35

    def extract_region(self, mask):
        """Retourne les 78 features pour un masque pixel arbitraire."""
        if mask.sum() < 3:
            return np.zeros(78, dtype=np.float32)
        return np.concatenate([
            self._color(mask),    # 16
            self._texture(mask),  # 15
            self._location(mask), # 12
            self._geometry(mask), # 35
        ])                        # = 78


# ══════════════════════════════════════════════════════════════════════════════
# Étape A — Modèle d'affinité
# ══════════════════════════════════════════════════════════════════════════════

def sample_affinity_pairs(ds, train_indices, rng):
    """
    Échantillonne N_PAIRS_EACH paires same-label et N_PAIRS_EACH paires diff-label.
    Input features de chaque paire : |x_i - x_j|  (dim 78).
    """
    print("  collecting superpixel features from training images...")
    all_feats, all_labels = [], []
    for idx in train_indices:
        _, _, meta = ds[idx]
        ft = load_ft(meta["imname"])
        if ft is None:
            continue
        sp_ids = ft.get("sp_ids", np.arange(len(ft["features"])))
        sp_gt  = ft["sp_labels"][sp_ids]          # GT label indexé par feat index
        labeled = sp_gt > 0
        all_feats.append(ft["features"][labeled].astype(np.float32))
        all_labels.append(sp_gt[labeled])

    all_feats  = np.vstack(all_feats)
    all_labels = np.concatenate(all_labels)
    n          = len(all_feats)
    print(f"  {n} labeled superpixels from {len(train_indices)} images")

    # Index par classe
    cls_idx = {l: np.where(all_labels == l)[0] for l in LABEL_IDS}

    # Same-label pairs — échantillonnage équilibré par classe
    same_X = []
    for l in LABEL_IDS:
        idx = cls_idx[l]
        if len(idx) < 2:
            continue
        n_l = N_PAIRS_EACH // len(LABEL_IDS)
        i   = rng.choice(idx, size=n_l)
        j   = rng.choice(idx, size=n_l)
        same_X.append(np.abs(all_feats[i] - all_feats[j]))
    same_X = np.vstack(same_X)[:N_PAIRS_EACH]

    # Diff-label pairs — toutes les paires de classes
    diff_X  = []
    pairs_c = [(1,2), (1,3), (2,3)]
    n_each  = N_PAIRS_EACH // len(pairs_c)
    for l1, l2 in pairs_c:
        idx1, idx2 = cls_idx[l1], cls_idx[l2]
        if not len(idx1) or not len(idx2):
            continue
        i = rng.choice(idx1, size=n_each)
        j = rng.choice(idx2, size=n_each)
        diff_X.append(np.abs(all_feats[i] - all_feats[j]))
    diff_X = np.vstack(diff_X)[:N_PAIRS_EACH]

    X = np.vstack([same_X, diff_X])
    y = np.array([1]*len(same_X) + [0]*len(diff_X), dtype=np.int32)
    print(f"  pairs: {len(same_X)} same-label  +  {len(diff_X)} diff-label")
    return X, y


def train_affinity_model(X, y):
    clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=AFF_DEPTH),
        n_estimators=AFF_N_EST,
        random_state=42,
    )
    clf.fit(X, y)
    train_acc = (clf.predict(X) == y).mean()
    print(f"  affinity model trained  | train acc = {train_acc:.4f}")
    return clf


# ══════════════════════════════════════════════════════════════════════════════
# Étape B — Algorithme glouton (vectorisé)
# ══════════════════════════════════════════════════════════════════════════════

def compute_affinity_matrix(feats, affinity_clf):
    """
    Calcule la matrice N×N de log-odds d'affinité entre superpixels.
    Input :  feats (N, 78)
    Output : aff_matrix (N, N)  symétrique,  valeurs = log P(même) / P(diff)
    """
    n = len(feats)
    if n < 2:
        return np.zeros((n, n), dtype=np.float32)

    i_idx, j_idx = np.triu_indices(n, k=1)          # paires du triangle supérieur
    diffs = np.abs(feats[i_idx] - feats[j_idx])     # (n_pairs, 78)

    proba    = affinity_clf.predict_proba(diffs)     # (n_pairs, 2) : [P(diff), P(same)]
    log_odds = np.log((proba[:, 1] + 1e-8) / (proba[:, 0] + 1e-8))

    aff = np.zeros((n, n), dtype=np.float32)
    aff[i_idx, j_idx] = log_odds
    aff[j_idx, i_idx] = log_odds
    np.fill_diagonal(aff, 0.0)
    return aff


def greedy_group(aff_matrix, nr, rng):
    """
    Algorithme glouton Hoiem 2005 section 3.1 (vectorisé NumPy).

    1. Ordonner aléatoirement les superpixels
    2. Affecter les nr premiers à des régions distinctes (seeds)
    3. Assigner chaque SP restant à la région avec la max log-vraisemblance moyenne
    4. Répéter l'étape 3  N_GREEDY_IT fois

    Retourne : assignment (n_sp,) — région 0..nr-1 pour chaque SP (indexé par feat index)
    """
    n_sp = len(aff_matrix)
    nr   = min(nr, n_sp)
    order     = rng.permutation(n_sp)
    seed_sps  = order[:nr]
    remaining = order[nr:]

    assignment = np.full(n_sp, 0, dtype=np.int32)
    for k, s in enumerate(seed_sps):
        assignment[s] = k

    for _ in range(N_GREEDY_IT):
        # Matrice d'appartenance (nr, n_sp) : membership[k,j]=1 si j ∈ région k
        membership   = np.zeros((nr, n_sp), dtype=np.float32)
        for k in range(nr):
            in_k = assignment == k
            if in_k.any():
                membership[k, in_k] = 1.0

        region_sizes = np.maximum(membership.sum(axis=1, keepdims=True), 1.0)

        if len(remaining) == 0:
            break

        # scores (n_remaining, nr) = affinité moyenne avec chaque région
        aff_sub = aff_matrix[remaining]                        # (n_rem, n_sp)
        scores  = (aff_sub @ membership.T) / region_sizes.T   # (n_rem, nr)
        assignment[remaining] = scores.argmax(axis=1)

    return assignment


# ══════════════════════════════════════════════════════════════════════════════
# Étape C — GT régions + homogénéité + features de régions
# ══════════════════════════════════════════════════════════════════════════════

def compute_region_gt(assignment, sp_gt_feat, sp_pixel_counts_feat, nr):
    """
    Calcule le label majoritaire et l'homogénéité de chaque région.
    Homogénéité : ≤ HOMOG_TOL (5 %) de pixels non-majoritaires (footnote 3).

    sp_gt_feat            : (n_sp,) GT label par feat index (0=non labellisé)
    sp_pixel_counts_feat  : (n_sp,) nombre de pixels par superpixel (feat index)
    """
    region_labels = np.zeros(nr, dtype=np.int32)
    region_homog  = np.zeros(nr, dtype=bool)

    for k in range(nr):
        members  = np.where(assignment == k)[0]
        if not len(members):
            continue
        pix_cts  = sp_pixel_counts_feat[members]
        labels   = sp_gt_feat[members]

        # ne garder que les SP labellisés
        lmask    = labels > 0
        if not lmask.any():
            continue

        lbl_pix  = pix_cts[lmask]
        lbl_lbl  = labels[lmask]
        total    = lbl_pix.sum()

        # label majoritaire (en pixels)
        best_l, best_n = 0, 0
        for l in LABEL_IDS:
            n_l = lbl_pix[lbl_lbl == l].sum()
            if n_l > best_n:
                best_n, best_l = n_l, l

        majority_frac         = best_n / (total + 1e-8)
        region_labels[k]      = best_l
        region_homog[k]       = (1 - majority_frac) <= HOMOG_TOL

    return region_labels, region_homog


def compute_region_features(extractor, segments, sp_ids_array,
                             assignment, nr):
    """
    Calcule les 78 features sur le masque pixel de chaque région.

    extractor    : FeatureExtractor déjà initialisé pour l'image
    segments     : (H, W) tableau de SP IDs
    sp_ids_array : (n_sp,) SP IDs triés (feat index → SP ID dans segments)
    assignment   : (n_sp,) région k pour chaque feat index
    """
    reg_features   = np.zeros((nr, 78), dtype=np.float32)
    reg_pixel_sizes = np.zeros(nr, dtype=np.int32)

    for k in range(nr):
        feat_members  = np.where(assignment == k)[0]
        if not len(feat_members):
            continue
        actual_sp_ids = sp_ids_array[feat_members]           # vrais IDs dans segments
        region_mask   = np.isin(segments, actual_sp_ids)
        n_pix         = int(region_mask.sum())
        reg_pixel_sizes[k] = n_pix
        if n_pix >= 3:
            reg_features[k] = extractor.extract_region(region_mask)

    return reg_features, reg_pixel_sizes


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    rng = np.random.default_rng(42)

    ds = HoiemDataset(root_dir=DATASET_DIR)
    train_ds, _ = ds.get_split()
    train_indices = train_ds._i
    print(f"Dataset: {len(ds)} images  |  train (cluster_images): {len(train_indices)}")

    # ── Étape A : entraîner le modèle d'affinité ──────────────────────────────
    aff_model_path = MODEL_DIR / "affinity_adaboost.pkl"
    if aff_model_path.exists():
        print("\nModèle d'affinité déjà présent — chargement...")
        with open(aff_model_path, "rb") as f:
            affinity_clf = pickle.load(f)
    else:
        print("\nÉtape A : entraînement du modèle d'affinité...")
        X_aff, y_aff = sample_affinity_pairs(ds, train_indices, rng)
        affinity_clf  = train_affinity_model(X_aff, y_aff)
        with open(aff_model_path, "wb") as f:
            pickle.dump(affinity_clf, f)
        print(f"  → sauvegardé dans {aff_model_path}")

    # ── Étapes B+C : générer les hypothèses pour toutes les images ────────────
    print(f"\nÉtapes B+C : hypothèses pour {len(ds)} images "
          f"(nr ∈ {NR_VALUES}, {N_GREEDY_IT} itérations glouton)...")

    skipped, done, cached = 0, 0, 0
    for idx in tqdm(range(len(ds)), desc="hypotheses"):
        _, label_map, meta = ds[idx]
        stem     = Path(meta["imname"]).stem
        out_path = HYPO_CACHE / f"{stem}.npy"

        if out_path.exists():
            cached += 1
            continue

        ft = load_ft(meta["imname"])
        sp = load_sp(meta["imname"])
        if ft is None or sp is None:
            skipped += 1
            continue

        feats       = ft["features"].astype(np.float32)       # (n_sp, 78)
        sp_ids_arr  = ft.get("sp_ids",
                             np.unique(sp["segments"]))        # (n_sp,) SP IDs réels
        sp_ids_arr  = np.asarray(sp_ids_arr)
        sp_gt_sp    = ft["sp_labels"]                          # indexé par vrai SP ID
        sp_gt_feat  = sp_gt_sp[sp_ids_arr]                    # indexé par feat index

        segments    = sp["segments"]
        n_sp        = len(feats)

        # Compte pixels par superpixel (indexé par feat index)
        full_counts        = np.bincount(segments.flatten(),
                                         minlength=int(segments.max())+1)
        sp_pixel_counts_feat = full_counts[sp_ids_arr]

        # Matrice d'affinité (n_sp × n_sp)
        aff_matrix = compute_affinity_matrix(feats, affinity_clf)

        # Extracteur de features (lignes Hough calculées une seule fois)
        image     = ds._load_image(ds._img_paths[stem.lower()])
        extractor = FeatureExtractor(image)

        # Générer une hypothèse par valeur de nr
        hypotheses = []
        for nr in NR_VALUES:
            assignment  = greedy_group(aff_matrix, nr, rng)
            nr_actual   = int(assignment.max()) + 1

            reg_labels, reg_homog = compute_region_gt(
                assignment, sp_gt_feat, sp_pixel_counts_feat, nr_actual)

            reg_features, reg_pixel_sizes = compute_region_features(
                extractor, segments, sp_ids_arr, assignment, nr_actual)

            hypotheses.append({
                "nr"               : nr,
                "n_regions"        : nr_actual,
                "assignment"       : assignment,           # (n_sp,) feat-indexed
                "region_features"  : reg_features,        # (nr_actual, 78)
                "region_labels"    : reg_labels,          # (nr_actual,)
                "region_homogeneous": reg_homog,          # (nr_actual,) bool
                "region_pixel_sizes": reg_pixel_sizes,   # (nr_actual,) pixels
            })

        np.save(out_path, {
            "imname"    : meta["imname"],
            "n_sp"      : n_sp,
            "sp_ids"    : sp_ids_arr,
            "hypotheses": hypotheses,
        })
        done += 1

    print(f"\nTerminé : {done} générés  |  {cached} déjà en cache  |  {skipped} ignorés")
    print(f"→ {HYPO_CACHE}/  ({len(list(HYPO_CACHE.glob('*.npy')))} fichiers)")

    # ── Vérification rapide sur la première image ──────────────────────────────
    _, _, meta0 = ds[0]
    hypo_data   = np.load(HYPO_CACHE / f"{Path(meta0['imname']).stem}.npy",
                          allow_pickle=True).item()
    print(f"\nVérification image 0 : {meta0['imname']}")
    for h in hypo_data["hypotheses"]:
        homog_frac = h["region_homogeneous"].mean()
        print(f"  nr={h['nr']:2d}  n_regions={h['n_regions']:2d}  "
              f"homog_frac={homog_frac:.2f}  "
              f"labels={np.unique(h['region_labels'])}")

    # ── Statistiques globales ─────────────────────────────────────────────────
    print("\nStatistiques globales sur les hypothèses générées :")
    total_homog, total_regions = 0, 0
    for p in list(HYPO_CACHE.glob("*.npy"))[:50]:
        d = np.load(p, allow_pickle=True).item()
        for h in d["hypotheses"]:
            total_homog   += h["region_homogeneous"].sum()
            total_regions += h["n_regions"]
    if total_regions:
        print(f"  % régions homogènes (50 premières images) : "
              f"{total_homog/total_regions*100:.1f}%  "
              f"(papier : ~40 %)")

    print("\ndone → data/hypotheses/  data/models/affinity_adaboost.pkl")


if __name__ == "__main__":
    main()
