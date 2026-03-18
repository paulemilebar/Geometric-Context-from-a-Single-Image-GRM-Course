import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.segmentation import mark_boundaries

sys.path.insert(0, "src")
from dataset import HoiemDataset, LABEL_COLORS
from superpixels import (compute_superpixels, build_adjacency, sp_gt_labels,
                         boundary_recall, save_cache, load_sp, DEFAULT_PARAMS)

DATASET_DIR = Path("dataset")
OUT_DIR     = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)

PARAM_GRID = [
    dict(scale=50,  sigma=0.5, min_size=50),
    dict(scale=100, sigma=0.8, min_size=100),  # hoiem params
    dict(scale=200, sigma=0.8, min_size=100),
    dict(scale=500, sigma=1.0, min_size=200),
]
PARAM_LABELS = ["fine (50)", "hoiem (100)*", "medium (200)", "coarse (500)"]


def underseg_error(segments, label_map):
    total = label_map.size
    err   = 0
    for gt_l in np.unique(label_map[label_map > 0]):
        gt_mask = label_map == gt_l
        for sp_id in np.unique(segments[gt_mask]):
            err += (segments == sp_id & ~gt_mask).sum()
    return err / total


def colored_sp(image, segments, label_map):
    lbl = sp_gt_labels(segments, label_map)
    out = np.zeros_like(image)
    for sp_id in range(segments.max() + 1):
        out[segments == sp_id] = LABEL_COLORS.get(int(lbl[sp_id]), (128,128,128))
    return out


def main():
    ds = HoiemDataset(root_dir=DATASET_DIR)

    image, label_map, meta = ds[0]
    segs = compute_superpixels(image, **DEFAULT_PARAMS)
    br   = boundary_recall(segs, label_map)
    print(f"{meta['imname']} | {segs.max()+1} superpixels | BR={br:.3f}")

    # single image visualization
    bnd = (mark_boundaries(image, segs, color=(1,1,0)) * 255).astype(np.uint8)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(f"{meta['imname']} - hoiem params")
    axes[0].imshow(image);             axes[0].set_title("image");          axes[0].axis("off")
    axes[1].imshow(bnd);               axes[1].set_title(f"{segs.max()+1} superpixels"); axes[1].axis("off")
    gt_rgb = np.zeros((*label_map.shape, 3), dtype=np.uint8)
    for lid, c in LABEL_COLORS.items():
        gt_rgb[label_map == lid] = c
    axes[2].imshow(gt_rgb);            axes[2].set_title("ground truth");   axes[2].axis("off")
    axes[3].imshow(colored_sp(image, segs, label_map))
    axes[3].set_title("sp colored by gt"); axes[3].axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "step2_example.png", dpi=150, bbox_inches="tight")
    plt.close()

    # param comparison
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("felzenszwalb param comparison", fontsize=13)
    for col, (p, lbl) in enumerate(zip(PARAM_GRID, PARAM_LABELS)):
        s  = compute_superpixels(image, **p)
        br = boundary_recall(s, label_map)
        bnd_ = (mark_boundaries(image, s, color=(1,1,0)) * 255).astype(np.uint8)
        axes[0, col].imshow(bnd_)
        axes[0, col].set_title(f"{lbl}\n{s.max()+1} sp | BR={br:.3f}", fontsize=9)
        axes[0, col].axis("off")
        axes[1, col].imshow(colored_sp(image, s, label_map))
        axes[1, col].axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "step2_params.png", dpi=150, bbox_inches="tight")
    plt.close()

    # overview on 6 images
    indices = [0, 30, 60, 90, 150, 250]
    fig, axes = plt.subplots(3, 6, figsize=(24, 12))
    for col, idx in enumerate(indices):
        img, lm, meta = ds[idx]
        s   = compute_superpixels(img, **DEFAULT_PARAMS)
        br  = boundary_recall(s, lm)
        bnd_ = (mark_boundaries(img, s, color=(1,1,0)) * 255).astype(np.uint8)
        gt  = np.zeros((*lm.shape, 3), dtype=np.uint8)
        for lid, c in LABEL_COLORS.items():
            gt[lm == lid] = c
        axes[0, col].imshow(img);   axes[0, col].set_title(Path(meta["imname"]).stem, fontsize=7)
        axes[1, col].imshow(bnd_);  axes[1, col].set_title(f"{s.max()+1} sp BR={br:.2f}", fontsize=7)
        axes[2, col].imshow(gt)
        for row in range(3): axes[row, col].axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "step2_overview.png", dpi=120, bbox_inches="tight")
    plt.close()

    # compute and cache all
    save_cache(ds)

    # metrics distribution
    recalls, n_sps = [], []
    for idx in range(len(ds)):
        _, lm, meta = ds[idx]
        cached = load_sp(meta["imname"])
        if cached:
            recalls.append(boundary_recall(cached["segments"], lm))
            n_sps.append(cached["n_sp"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(recalls, bins=20, color="#2ecc71", edgecolor="white")
    axes[0].axvline(np.mean(recalls), color="red", ls="--", label=f"mean={np.mean(recalls):.3f}")
    axes[0].set_title("boundary recall"); axes[0].legend()
    axes[1].hist(n_sps, bins=20, color="#3498db", edgecolor="white")
    axes[1].axvline(np.mean(n_sps), color="red", ls="--", label=f"mean={np.mean(n_sps):.0f}")
    axes[1].set_title("n superpixels"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "step2_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"mean BR={np.mean(recalls):.3f}  mean n_sp={np.mean(n_sps):.0f}")
    print("done -> outputs/step2_*.png  data/superpixels/")


if __name__ == "__main__":
    main()
