import numpy as np
import cv2
from pathlib import Path
from skimage.segmentation import felzenszwalb


SP_CACHE  = Path("data/superpixels")
FT_CACHE  = Path("data/features")

DEFAULT_PARAMS = dict(scale=100, sigma=0.8, min_size=100)


def compute_superpixels(image, scale=100, sigma=0.8, min_size=100):
    segs = felzenszwalb(image, scale=scale, sigma=sigma, min_size=min_size)
    return segs.astype(np.int32)


def build_adjacency(segments):
    n   = segments.max() + 1
    adj = np.zeros((n, n), dtype=np.uint8)

    l, r = segments[:, :-1].flatten(), segments[:, 1:].flatten()
    mask = l != r
    adj[l[mask], r[mask]] = 1; adj[r[mask], l[mask]] = 1

    t, b = segments[:-1, :].flatten(), segments[1:, :].flatten()
    mask = t != b
    adj[t[mask], b[mask]] = 1; adj[b[mask], t[mask]] = 1

    return adj


def sp_gt_labels(segments, label_map):
    n  = segments.max() + 1
    out = np.zeros(n, dtype=np.int32)
    for sp_id in range(n):
        mask = segments == sp_id
        if not mask.any():
            continue
        vals = label_map[mask]
        vals = vals[vals > 0]
        if len(vals):
            counts = np.bincount(vals)
            out[sp_id] = counts.argmax()
    return out


def boundary_recall(segments, label_map, tol=2):
    from scipy.ndimage import binary_dilation

    def edges(lmap):
        e = np.zeros(lmap.shape, bool)
        hd = np.diff(lmap, axis=0) != 0
        vd = np.diff(lmap, axis=1) != 0
        e[:-1] |= hd; e[1:] |= hd
        e[:, :-1] |= vd; e[:, 1:] |= vd
        return e

    gt_e = edges(label_map)
    sp_e = binary_dilation(edges(segments), iterations=tol)
    total = gt_e.sum()
    return float((gt_e & sp_e).sum() / total) if total else 1.0


def save_cache(ds, cache_dir=SP_CACHE, params=None):
    from tqdm import tqdm
    params = params or DEFAULT_PARAMS
    cache_dir.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(range(len(ds)), desc="superpixels"):
        image, label_map, meta = ds[idx]
        stem = Path(meta["imname"]).stem
        out  = cache_dir / f"{stem}.npy"
        if out.exists():
            continue

        segs = compute_superpixels(image, **params)
        adj  = build_adjacency(segs)
        lbl  = sp_gt_labels(segs, label_map)
        np.save(out, {"segments": segs, "adj": adj, "sp_labels": lbl,
                      "n_sp": segs.max()+1, "imname": meta["imname"]})


def load_sp(imname, cache_dir=SP_CACHE):
    p = cache_dir / f"{Path(imname).stem}.npy"
    return np.load(p, allow_pickle=True).item() if p.exists() else None


def load_ft(imname, cache_dir=FT_CACHE):
    p = cache_dir / f"{Path(imname).stem}.npy"
    return np.load(p, allow_pickle=True).item() if p.exists() else None
