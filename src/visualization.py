import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from skimage.segmentation import mark_boundaries

import sys
sys.path.insert(0, "src")

from superpixels import load_sp, load_ft
from dataset import HoiemDataset


SP_DIR  = Path("data/superpixels")
OUT_DIR = Path("outputs/graph_visu")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# FUNCTIONS ALREADY CODED IN BP
def compute_pairwise_weights(adjacency, features, beta=0.1):
    weights = {}

    for i in adjacency:
        for j in adjacency[i]:
            diff = np.linalg.norm(features[i] - features[j])
            w = np.exp(-beta * diff**2)

            weights[(i, j)] = w
            weights[(j, i)] = w

    return weights

def to_adj_dict(adjacency):
    if isinstance(adjacency, dict):
        return {int(i): set(map(int, neigh)) for i, neigh in adjacency.items()}
    
    if isinstance(adjacency, np.ndarray):
        N = adjacency.shape[0]
        adj = {}
        for i in range(N):
            neighbors = np.where(adjacency[i] > 0)[0]
            adj[i] = set(map(int, neighbors))
        return adj

    raise TypeError("Unsupported adjacency type")


# GRAPH VISUALIZATION
def compute_centers(segments):
    n_sp = segments.max() + 1
    centers = np.zeros((n_sp, 2))

    for sp_id in range(n_sp):
        ys, xs = np.where(segments == sp_id)
        if len(xs) == 0:
            continue
        centers[sp_id] = [xs.mean(), ys.mean()]

    return centers


def plot_superpixels(image, segments, save_path):
    bnd = (mark_boundaries(image, segments, color=(1,1,0)) * 255).astype(np.uint8)

    plt.figure(figsize=(6,6))
    plt.imshow(bnd)
    plt.title(f"{segments.max()+1} superpixels")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_graph(image, segments, adjacency, save_path):
    centers = compute_centers(segments)
    n_sp = segments.max() + 1

    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(image)

    for i in range(n_sp):
        for j in range(n_sp):
            if adjacency[i, j]:
                x1, y1 = centers[i]
                x2, y2 = centers[j]
                ax.plot([x1, x2], [y1, y2],
                        color="red", linewidth=0.5, alpha=0.4)

    ax.scatter(centers[:,0], centers[:,1], s=5, c="yellow")

    ax.set_title("Superpixel Graph")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_weighted_graph(image, segments, adjacency, weights, save_path):
    centers = compute_centers(segments)
    n_sp = segments.max() + 1

    fig, ax = plt.subplots(figsize=(6,6))
    ax.imshow(image)

    for i in range(n_sp):
        for j in range(n_sp):
            if adjacency[i, j]:
                w = weights.get((i, j), 0)
                lw = 0.5 + 3 * w

                x1, y1 = centers[i]
                x2, y2 = centers[j]

                ax.plot([x1, x2], [y1, y2],
                        color="red", linewidth=lw, alpha=0.5)

    ax.scatter(centers[:,0], centers[:,1], s=5, c="yellow")

    ax.set_title("Weighted Graph (pairwise)")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# MAIN LOOP
def main():
    files = sorted(SP_DIR.glob("*.npy"))

    print(f"Found {len(files)} superpixel files")

    # DATASET
    ds = HoiemDataset(root_dir="dataset")

    # mapping nom -> index dataset
    name_to_idx = {}
    for idx in range(len(ds)):
        _, _, meta = ds[idx]
        stem = Path(meta["imname"]).stem
        name_to_idx[stem] = idx

    for i, fpath in enumerate(files[:10]):  # limit of 10 images
        name = fpath.stem
        print(f"[{i}] Processing {name}")

        sp = load_sp(name)
        if sp is None:
            print("  -> skip (no superpixels)")
            continue

        ft = load_ft(name)
        if ft is None:
            print("  -> skip (no features)")
            continue

        if name not in name_to_idx:
            print("  -> skip (not in dataset)")
            continue

        # retrieve images from dataset
        idx = name_to_idx[name]
        image, _, _ = ds[idx]

        segments = sp["segments"]
        adjacency = sp["adj"]

        # weights
        adjacency_dict = to_adj_dict(adjacency)
        weights = compute_pairwise_weights(adjacency_dict, ft["features"])

        # outputs
        plot_superpixels(image, segments,
                         OUT_DIR / f"{name}_superpixels.png")

        plot_graph(image, segments, adjacency,
                   OUT_DIR / f"{name}_graph.png")

        plot_weighted_graph(image, segments, adjacency, weights,
                            OUT_DIR / f"{name}_weighted.png")

    print(f"\nDone -> see {OUT_DIR}")


if __name__ == "__main__":
    main()