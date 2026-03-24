import sys
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial import ConvexHull
from scipy.stats import entropy
from tqdm import tqdm

sys.path.insert(0, "src")
from dataset import HoiemDataset
from superpixels import load_sp

DATASET_DIR = Path("dataset")
SP_CACHE    = Path("data/superpixels")
FT_CACHE    = Path("data/features"); FT_CACHE.mkdir(parents=True, exist_ok=True)
OUT_DIR     = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)

HORIZON_RATIO  = 0.40
N_DOOG         = 12
DOOG_ANGLES    = np.linspace(0, 180, N_DOOG, endpoint=False)

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
assert len(FEATURE_NAMES) == 78

GROUPS = {"color": (0,16), "location": (16,28), "texture": (28,43), "geometry": (43,78)}


def _gauss2d(sigma, size):
    ax = np.arange(-size//2+1, size//2+1, dtype=np.float64)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2+yy**2)/(2*sigma**2))
    return k / k.sum()


def build_doog_filters(sigma1=1.0, sigma2=3.0, size=15):
    base = _gauss2d(sigma1, size) - _gauss2d(sigma2, size)
    filters = []
    for deg in DOOG_ANGLES:
        M = cv2.getRotationMatrix2D((size//2, size//2), deg, 1.0)
        filters.append(cv2.warpAffine(base.astype(np.float32), M, (size, size)))
    return filters

DOOG_FILTERS = build_doog_filters()


def apply_doog(gray):
    gf = gray.astype(np.float32) / 255.0
    out = np.zeros((*gray.shape, N_DOOG), dtype=np.float32)
    for i, f in enumerate(DOOG_FILTERS):
        out[..., i] = np.abs(cv2.filter2D(gf, -1, f))
    return out


class FeatureExtractor:
    def __init__(self, image):
        self.H, self.W = image.shape[:2]
        self.rgb  = image.astype(np.float32) / 255.0
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        self.hsv  = hsv / np.array([180., 255., 255.])
        self.gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.doog = apply_doog(self.gray)
        self.horizon_y = self.H * HORIZON_RATIO
        self._lines = None

    def _color(self, mask):
        f = []
        f += self.rgb[mask].mean(axis=0).tolist()                              # c1 (3)
        f += self.hsv[mask].mean(axis=0).tolist()                              # c2 (3)
        h  = self.hsv[mask, 0]
        hh, _ = np.histogram(h, bins=5, range=(0,1), density=True)
        hh /= hh.sum() + 1e-8
        f += hh.tolist() + [float(entropy(hh+1e-8))]                          # c3 (6)
        s  = self.hsv[mask, 1]
        sh, _ = np.histogram(s, bins=3, range=(0,1), density=True)
        sh /= sh.sum() + 1e-8
        f += sh.tolist() + [float(entropy(sh+1e-8))]                          # c4 (4)
        return np.array(f, dtype=np.float32)

    def _texture(self, mask):
        t1 = self.doog[mask].mean(axis=0)                                     # (12,)
        return np.array([*t1,
                         t1.mean(),
                         t1.argmax() / 11.0,
                         t1.max() - np.median(t1)], dtype=np.float32)

    def _location(self, mask):
        ys, xs = np.where(mask)
        xn, yn = xs / self.W, ys / self.H
        hn = self.horizon_y / self.H
        hull_sides, compact = 4.0, 0.5
        if len(ys) >= 4:
            try:
                hull = ConvexHull(np.column_stack([xs, ys]).astype(np.float64))
                hull_sides = float(len(hull.vertices))
                compact    = float(np.clip(mask.sum() / (hull.volume + 1e-8), 0, 1))
            except Exception:
                pass
        n_sp_norm = float(len(ys)) / (self.H * self.W)  # l4: normalized superpixel area
        x_range   = float(xs.max() - xs.min()) / self.W if len(xs) > 1 else 0.0  # l7: x-spread proxy
        return np.array([xn.mean(), yn.mean(),
                         np.percentile(xn,10), np.percentile(xn,90),
                         np.percentile(yn,10), np.percentile(yn,90),
                         np.percentile(yn,10)-hn, np.percentile(yn,90)-hn,
                         n_sp_norm, hull_sides, compact, x_range], dtype=np.float32)

    def _get_lines(self):
        if self._lines is not None:
            return self._lines
        edges = cv2.Canny(self.gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        self._lines = lines[:, 0, :] if lines is not None else np.zeros((0,4), np.float32)
        return self._lines

    def _lines_in_mask(self, lines, mask):
        if not len(lines): return lines
        out = []
        for x1,y1,x2,y2 in lines:
            x1c,y1c = int(np.clip(x1,0,self.W-1)), int(np.clip(y1,0,self.H-1))
            x2c,y2c = int(np.clip(x2,0,self.W-1)), int(np.clip(y2,0,self.H-1))
            if mask[y1c,x1c] or mask[y2c,x2c]:
                out.append([x1,y1,x2,y2])
        return np.array(out, np.float32) if out else np.zeros((0,4))

    @staticmethod
    def _angle(x1,y1,x2,y2): return np.arctan2(y2-y1, x2-x1) % np.pi

    def _parallel_frac(self, lines, tol=np.pi/8):
        if len(lines) < 2: return 0.0
        angs = [self._angle(*l) for l in lines]
        total = pairs = 0
        for i in range(len(angs)):
            for j in range(i+1, len(angs)):
                d = abs(angs[i]-angs[j]); d = min(d, np.pi-d)
                total += 1
                if d < tol: pairs += 1
        return pairs / total if total else 0.0

    @staticmethod
    def _intersect(l1, l2):
        x1,y1,x2,y2 = l1; x3,y3,x4,y4 = l2
        d = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(d) < 1e-8: return None
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / d
        return (x1+t*(x2-x1), y1+t*(y2-y1))

    def _intersections(self, lines):
        if len(lines) < 2: return np.zeros((0,2))
        pts = [self._intersect(lines[i], lines[j])
               for i in range(len(lines)) for j in range(i+1, len(lines))]
        pts = [p for p in pts if p]
        return np.array(pts, np.float32) if pts else np.zeros((0,2))

    def _far_frac(self, pts, cx, cy, thr):
        out = np.zeros(8, np.float32)
        if not len(pts): return out
        for k in range(8):
            a0, a1 = k*np.pi/4, (k+1)*np.pi/4
            angles = np.arctan2(pts[:,1]-cy, pts[:,0]-cx) % (2*np.pi)
            dists  = np.hypot(pts[:,0]-cx, pts[:,1]-cy)
            sect   = (angles >= a0) & (angles < a1)
            if sect.any(): out[k] = float((dists[sect] > thr).mean())
        return out

    def _geometry(self, mask):
        lines = self._get_lines()
        lm    = self._lines_in_mask(lines, mask)
        cx, cy = self.W/2, self.H/2
        diag   = np.hypot(self.W, self.H)

        g1 = float(len(lm)) / max(1, mask.sum()/100)
        g2 = self._parallel_frac(lm)

        pts = self._intersections(lm)
        if len(pts):
            angs = np.degrees(np.arctan2(pts[:,1]-cy, pts[:,0]-cx)) % 360
            hist, _ = np.histogram(angs, bins=12, range=(0,360))
            hn = hist.astype(np.float32); hn /= hn.sum()+1e-8
            g3 = np.append(hn, entropy(hn+1e-8))
            g4 = float((pts[:,0] > cx).mean())
            g5 = float((pts[:,1] < cy).mean())
        else:
            g3 = np.zeros(13, np.float32)
            g4 = g5 = 0.5

        g6 = self._far_frac(pts, cx, cy, 1.5*diag)
        g7 = self._far_frac(pts, cx, cy, 5.0*diag)

        ys, xs = np.where(mask)
        if len(ys) >= 4:
            resp = self.doog[ys, xs].mean(axis=1)
            xn, yn = xs/self.W, ys/self.H
            gx = float(np.corrcoef(xn, resp)[0,1]) if xn.std()>1e-8 else 0.0
            gy = float(np.corrcoef(yn, resp)[0,1]) if yn.std()>1e-8 else 0.0
            gx = 0.0 if np.isnan(gx) else gx
            gy = 0.0 if np.isnan(gy) else gy
        else:
            gx = gy = 0.0

        return np.concatenate([[g1, g2], g3, [g4, g5], g6, g7, [gx, gy]]).astype(np.float32)

    def extract_all(self, segments):
        sp_ids = np.unique(segments)
        feats  = np.zeros((len(sp_ids), 78), dtype=np.float32)
        for i, sp_id in enumerate(sp_ids):
            mask = segments == sp_id
            if mask.sum() < 3: continue
            feats[i] = np.concatenate([self._color(mask), self._texture(mask),
                                        self._location(mask), self._geometry(mask)])
        return feats


def extract_and_cache(ds):
    for idx in tqdm(range(len(ds)), desc="features"):
        _, label_map, meta = ds[idx]
        stem = Path(meta["imname"]).stem
        out  = FT_CACHE / f"{stem}.npy"
        if out.exists(): continue

        sp = load_sp(meta["imname"])
        if sp is None:
            print(f"no superpixels for {meta['imname']}, skip")
            continue

        image = ds._load_image(ds._img_paths[stem.lower()])
        feats = FeatureExtractor(image).extract_all(sp["segments"])

        np.save(out, {"features": feats, "sp_labels": sp["sp_labels"],
                      "sp_ids": np.unique(sp["segments"]), "imname": meta["imname"]})


def main():
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = HoiemDataset(root_dir=DATASET_DIR)

    # test on first image
    image, label_map, meta = ds[0]
    sp   = load_sp(meta["imname"])
    segs = sp["segments"]

    feats = FeatureExtractor(image).extract_all(segs)
    print(f"{meta['imname']}: {feats.shape}  nan={np.isnan(feats).sum()}")
    for name, (a, b) in GROUPS.items():
        fg = feats[:, a:b]
        print(f"  {name:10s}: mean={fg.mean():.3f}  std={fg.std():.3f}  range=[{fg.min():.3f}, {fg.max():.3f}]")

    # feature distributions per class
    sp_labels = sp["sp_labels"]
    cls_names  = {1: "ground", 2: "vertical", 3: "sky"}
    cls_colors = {1: "#2ecc71", 2: "#e74c3c", 3: "#3498db"}
    plot_feats = ["l1_y", "c2_h", "c2_s", "t2_mean", "t4_contrast",
                  "l6_compact", "g1_lines", "g2_parallel", "g3_ent",
                  "g4_right", "g5_above", "g8_gx"]

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    fig.suptitle("feature distributions per geometric class", fontsize=13)
    for ax, fname in zip(axes.flat, plot_feats):
        fidx = FEATURE_NAMES.index(fname)
        for lid, lname in cls_names.items():
            vals = feats[sp_labels == lid, fidx]
            if len(vals): ax.hist(vals, bins=20, alpha=0.6, label=lname,
                                  color=cls_colors[lid], density=True)
        ax.set_title(fname, fontsize=8); ax.legend(fontsize=6); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(OUT_DIR / "step3_feature_dists.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("saved outputs/step3_feature_dists.png")

    extract_and_cache(ds)
    print(f"done -> {FT_CACHE}/ ({len(list(FT_CACHE.glob('*.npy')))} files)")


if __name__ == "__main__":
    main()
