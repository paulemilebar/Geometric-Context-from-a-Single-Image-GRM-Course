import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, "src")
from dataset import HoiemDataset, LABEL_NAMES, LABEL_COLORS, LABEL_IDS

DATASET_DIR = Path("dataset")
OUT_DIR     = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)


def main():
    ds = HoiemDataset(root_dir=DATASET_DIR)

    # basic info on first entry
    ds.print_entry(0)

    # load one sample
    image, label_map, meta = ds[0]
    print(f"\nimage:     {image.shape}")
    print(f"label_map: {label_map.shape}")
    print(f"nseg:      {meta['nseg']}")
    total = label_map.size
    for lid in LABEL_IDS:
        print(f"  {LABEL_NAMES[lid]:10s}: {(label_map==lid).sum()/total*100:.1f}%")

    # split
    train, test = ds.get_split()
    print(f"\ntrain: {len(train)}  test: {len(test)}")
    img_tr, _, _ = train[0]
    img_te, _, _ = test[0]
    print(f"train[0]: {img_tr.shape}  test[0]: {img_te.shape}")

    # overview grid
    indices = [0, len(ds)//5, 2*len(ds)//5, 3*len(ds)//5, 4*len(ds)//5, len(ds)-1]
    fig, axes = plt.subplots(2, 6, figsize=(22, 8))
    fig.suptitle("hoiem dataset samples", fontsize=13)

    for col, idx in enumerate(indices):
        img, lm, meta = ds[idx]
        axes[0, col].imshow(img)
        axes[0, col].set_title(Path(meta["imname"]).stem, fontsize=7)
        axes[0, col].axis("off")
        axes[1, col].imshow(ds.colored(lm))
        axes[1, col].axis("off")

    patches = [mpatches.Patch(color=[c/255 for c in LABEL_COLORS[l]], label=LABEL_NAMES[l])
               for l in LABEL_IDS]
    fig.legend(handles=patches, loc="lower center", ncol=3)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(OUT_DIR / "step1_overview.png", dpi=150, bbox_inches="tight")
    plt.close()

    ds.visualize(0, save_path=str(OUT_DIR / "step1_sample.png"))
    print("done -> outputs/step1_*.png")


if __name__ == "__main__":
    main()
