import sys
import json
import csv
import pickle
from datetime import datetime
from pathlib import Path
from sklearn.base import clone
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset import HoiemDataset, LABEL_NAMES, LABEL_COLORS, LABEL_IDS
from superpixels import load_sp, load_ft, build_adjacency


DATASET_DIR = ROOT_DIR / "dataset"
HYPO_CACHE  = ROOT_DIR / "data" / "hypotheses"
MODEL_DIR   = ROOT_DIR / "data" / "models"
OUT_DIR     = ROOT_DIR / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# HYPERPARAMETERS
LAMBDA_PAIRWISE = 0.1
BETA = 0.05
BETA_COLOR = 0.2
BETA_TEXTURE = 0.1
BETA_POS = 0.05
N_ITERS = 15
MAX_IMAGES = 90


# HYPOTHESIS LOADING
def load_hypothesis(imname):
    p = HYPO_CACHE / f"{Path(imname).stem}.npy"
    return np.load(p, allow_pickle=True).item() if p.exists() else None


# MULTIH PROBAS
def predict_image_multiH_proba(hd, label_clf, homog_clf):
    n_sp = hd["n_sp"]
    confidence = np.zeros((n_sp, 3), dtype=np.float64)
    homog_sum = np.zeros(n_sp, dtype=np.float64)

    for h in hd["hypotheses"]:
        nr_actual = h["n_regions"]
        assignment = h["assignment"] 
        reg_feats = h["region_features"] 

        if nr_actual == 0:
            continue

        # P(label=v | region)
        label_proba = label_clf.predict_proba(reg_feats)   # (nr, 3)

        # P(region homogeneous)
        if hasattr(homog_clf, "classes_") and len(homog_clf.classes_) == 2:
            homog_proba = homog_clf.predict_proba(reg_feats)[:, 1]
        else:
            homog_proba = np.ones(nr_actual, dtype=np.float64)

        for k in range(nr_actual):
            sp_in_region = np.where(assignment == k)[0]
            if len(sp_in_region) == 0:
                continue

            confidence[sp_in_region] += label_proba[k] * homog_proba[k]
            homog_sum[sp_in_region] += homog_proba[k]

    confidence /= np.maximum(homog_sum[:, None], 1e-8)
    confidence /= np.maximum(confidence.sum(axis=1, keepdims=True), 1e-8)

    return confidence


# ENERGY
def compute_energy(labels, unary, adjacency, lam, weights):
    E = np.sum(unary[np.arange(len(labels)), labels])
    for i in range(len(labels)):
        for j in adjacency[i]:
            if i < j and labels[i] != labels[j]:
                w = weights.get((i, j), 1.0)
                E += lam * w
    return E


def compute_energy_full(labels, unary, pairwise, adjacency):
    E = 0.0

    for i in range(len(labels)):
        E += unary[i, labels[i]]

    for i in adjacency:
        for j in adjacency[i]:
            if i < j:
                E += pairwise[(i, j)][labels[i], labels[j]]

    return E


# ADJACENCY
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


# PAIRWISE WEIGHTS
def compute_pairwise_weights(adjacency, features, beta=BETA):
    weights = {}

    for i in adjacency:
        for j in adjacency[i]:
            diff = features[i] - features[j]
            d = np.dot(diff, diff)
            w = np.exp(-beta * d)
            weights[(i, j)] = w
            weights[(j, i)] = w

    return weights


def compute_pairwise_weights_color_texture_position(
    adjacency,
    features,
    beta_color=BETA_COLOR,
    beta_texture=BETA_TEXTURE,
    beta_pos=BETA_POS
):
    weights = {}

    color_idx = slice(0, 16)
    texture_idx = slice(16, 31)
    pos_idx = slice(31, 33)

    for i in adjacency:
        for j in adjacency[i]:
            diff_color = features[i][color_idx] - features[j][color_idx]
            d_color = np.dot(diff_color, diff_color)

            diff_texture = features[i][texture_idx] - features[j][texture_idx]
            d_texture = np.dot(diff_texture, diff_texture)

            diff_pos = features[i][pos_idx] - features[j][pos_idx]
            d_pos = np.dot(diff_pos, diff_pos)

            w = np.exp(
                - beta_color * d_color
                - beta_texture * d_texture
                - beta_pos * d_pos
            )

            weights[(i, j)] = w
            weights[(j, i)] = w

    return weights


# BELIEF PROPAGATION
def belief_propagation_reparametrization(
    unary_init,
    adjacency,
    weights,
    n_iters=20,
    lam=0.5,
    track_energy=False
):
    N, L = unary_init.shape

    adj = to_adj_dict(adjacency)
    for i in list(adj.keys()):
        for j in adj[i]:
            adj.setdefault(j, set()).add(i)
    adj = {i: list(v) for i, v in adj.items()}

    edges = []
    for i in adj:
        for j in adj[i]:
            if i < j:
                edges.append((i, j))

    unary = unary_init.copy()

    pairwise = {}
    for (i, j) in edges:
        w = weights.get((i, j), 1.0)

        V = np.full((L, L), lam * w)
        np.fill_diagonal(V, 0.0)

        pairwise[(i, j)] = V.copy()
        pairwise[(j, i)] = V.T.copy()

    energy_curve = []

    for _ in range(n_iters):
        # forward
        for (i, j) in edges:
            V = pairwise[(i, j)]

            M = np.zeros(L)
            for lj in range(L):
                M[lj] = np.min(unary[i] + V[:, lj])

            unary[j] += M
            pairwise[(i, j)] -= M[None, :]
            pairwise[(j, i)] = pairwise[(i, j)].T

            unary[j] -= unary[j].min()

        # backward
        for (i, j) in reversed(edges):
            V = pairwise[(j, i)]

            M = np.zeros(L)
            for li in range(L):
                M[li] = np.min(unary[j] + V[:, li])

            unary[i] += M
            pairwise[(j, i)] -= M[None, :]
            pairwise[(i, j)] = pairwise[(j, i)].T

            unary[i] -= unary[i].min()

        if track_energy:
            labels_tmp = np.argmin(unary, axis=1)

            E = 0.0
            for i in range(N):
                E += unary[i, labels_tmp[i]]
            for (i, j) in edges:
                E += pairwise[(i, j)][labels_tmp[i], labels_tmp[j]]

            energy_curve.append(E)

    labels = np.argmin(unary, axis=1)

    if track_energy:
        return labels, energy_curve
    return labels


# INFERENCE PER IMAGE
def run_bp_on_image(ft, sp, meta, label_clf, homog_clf, track_energy=False):
    feats = ft["features"]          # features superpixels for the pairwise
    segments = sp["segments"]

    hd = load_hypothesis(meta["imname"])
    if hd is None:
        raise ValueError(f"Hypothèses manquantes pour {meta['imname']}")

    # Unary MultiH at superpixel level
    probas = predict_image_multiH_proba(hd, label_clf, homog_clf)   # (n_sp, 3)
    unary = -np.log(probas + 1e-6)

    # Adjacence superpixels
    adj_matrix = build_adjacency(segments)
    adjacency = to_adj_dict(adj_matrix)

    # Pairwise on features superpixels
    # Variant 1
    weights = compute_pairwise_weights(adjacency, feats, beta=BETA)

    # Variant 2 if you want to "separate" color/texture/position
    # weights = compute_pairwise_weights_color_texture_position(
    #     adjacency, feats,
    #     beta_color=BETA_COLOR,
    #     beta_texture=BETA_TEXTURE,
    #     beta_pos=BETA_POS
    # )

    if track_energy:
        preds, energy_curve = belief_propagation_reparametrization(
            unary, adjacency, weights,
            n_iters=N_ITERS,
            lam=LAMBDA_PAIRWISE,
            track_energy=True
        )
    else:
        preds = belief_propagation_reparametrization(
            unary, adjacency, weights,
            n_iters=N_ITERS,
            lam=LAMBDA_PAIRWISE,
            track_energy=False
        )

    # Mapping preds {0,1,2} -> labels dataset {1,2,3}
    pred_map = np.zeros_like(segments, dtype=np.int32)

    sp_ids_arr = ft.get("sp_ids", np.unique(segments))
    sp_ids_arr = np.asarray(sp_ids_arr)

    for i, sp_id in enumerate(sp_ids_arr):
        if i < len(preds):
            pred_map[segments == sp_id] = preds[i] + 1

    if track_energy:
        return pred_map, energy_curve
    return pred_map


# METRICS
def pixel_accuracy_bp(ds, indices, label_clf, homog_clf):
    correct = 0
    total = 0

    for idx in indices:
        _, label_map, meta = ds[idx]
        ft = load_ft(meta["imname"])
        sp = load_sp(meta["imname"])

        if ft is None or sp is None:
            continue

        try:
            pred_map = run_bp_on_image(ft, sp, meta, label_clf, homog_clf)
        except ValueError:
            continue

        valid = label_map > 0
        correct += int((pred_map[valid] == label_map[valid]).sum())
        total += int(valid.sum())

    return correct / max(total, 1)


# PLOTS
def plot_confusion_bp(ds, indices, label_clf, homog_clf, save_path):
    all_preds, all_true = [], []

    for idx in indices[:50]:
        _, label_map, meta = ds[idx]
        ft = load_ft(meta["imname"])
        sp = load_sp(meta["imname"])

        if ft is None or sp is None:
            continue

        try:
            pred_map = run_bp_on_image(ft, sp, meta, label_clf, homog_clf)
        except ValueError:
            continue

        valid = label_map > 0
        all_true.extend(label_map[valid].tolist())
        all_preds.extend(pred_map[valid].tolist())

    cm = confusion_matrix(all_true, all_preds, labels=LABEL_IDS, normalize="true")

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(LABEL_IDS)))
    ax.set_yticks(range(len(LABEL_IDS)))
    ax.set_xticklabels([LABEL_NAMES[l] for l in LABEL_IDS])
    ax.set_yticklabels([LABEL_NAMES[l] for l in LABEL_IDS])

    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("BP confusion matrix (MultiH init)")

    for i in range(len(LABEL_IDS)):
        for j in range(len(LABEL_IDS)):
            ax.text(
                j, i, f"{cm[i, j]:.2f}",
                ha="center", va="center",
                color="white" if cm[i, j] > 0.5 else "black"
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_predictions_bp(ds, indices, label_clf, homog_clf, save_path, n=6):
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 12))
    fig.suptitle("BP predictions with MultiH unary init", fontsize=13)

    for col, idx in enumerate(indices[:n]):
        image, label_map, meta = ds[idx]
        ft = load_ft(meta["imname"])
        sp = load_sp(meta["imname"])

        if ft is None or sp is None:
            continue

        try:
            pred_map = run_bp_on_image(ft, sp, meta, label_clf, homog_clf)
        except ValueError:
            continue

        valid = label_map > 0
        acc = (pred_map[valid] == label_map[valid]).mean() if valid.any() else 0.0

        gt_rgb = np.zeros((*label_map.shape, 3), dtype=np.uint8)
        pred_rgb = np.zeros((*pred_map.shape, 3), dtype=np.uint8)

        for l, c in LABEL_COLORS.items():
            gt_rgb[label_map == l] = c
            pred_rgb[pred_map == l] = c

        axes[0, col].imshow(image)
        axes[0, col].set_title(Path(meta["imname"]).stem, fontsize=7)

        axes[1, col].imshow(gt_rgb)
        axes[1, col].set_title("GT", fontsize=7)

        axes[2, col].imshow(pred_rgb)
        axes[2, col].set_title(f"BP acc={acc:.2f}", fontsize=7)

        for r in range(3):
            axes[r, col].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def plot_energy_curves(ds, indices, label_clf, homog_clf, save_path, n=5):
    fig, ax = plt.subplots(figsize=(6, 4))

    for idx in indices[:n]:
        _, _, meta = ds[idx]
        ft = load_ft(meta["imname"])
        sp = load_sp(meta["imname"])

        if ft is None or sp is None:
            continue

        try:
            _, energy_curve = run_bp_on_image(
                ft, sp, meta, label_clf, homog_clf, track_energy=True
            )
        except ValueError:
            continue

        ax.plot(
            energy_curve,
            marker="o",
            markersize=2,
            label=Path(meta["imname"]).stem
        )

    ax.set_xlabel("iteration")
    ax.set_ylabel("energy")
    ax.set_title("BP energy convergence (MultiH init)")
    ax.legend(fontsize=6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# RESULTS SAVE
def save_results(results, out_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = out_dir / f"grid_search_{timestamp}.json"
    csv_path = out_dir / f"grid_search_{timestamp}.csv"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    if len(results) > 0:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r)

    print(f"\nResults saved to:\n{json_path}\n{csv_path}")


# GRID SEARCH
def hyperparameter_grid_search(
    ds,
    test_indices,
    label_clf,
    homog_clf,
    lambda_values,
    beta_values,
    max_images=MAX_IMAGES
):
    results = []

    for lam in lambda_values:
        for beta in beta_values:
            print(f"\nTesting lambda={lam:.3f}, beta={beta:.3f}")

            correct = 0
            total = 0

            for idx in test_indices[:max_images]:
                _, label_map, meta = ds[idx]
                ft = load_ft(meta["imname"])
                sp = load_sp(meta["imname"])

                if ft is None or sp is None:
                    continue

                hd = load_hypothesis(meta["imname"])
                if hd is None:
                    continue

                feats = ft["features"]
                segments = sp["segments"]

                probas = predict_image_multiH_proba(hd, label_clf, homog_clf)
                unary = -np.log(probas + 1e-6)

                adj_matrix = build_adjacency(segments)
                adjacency = to_adj_dict(adj_matrix)

                weights = compute_pairwise_weights(adjacency, feats, beta=beta)

                preds = belief_propagation_reparametrization(
                    unary, adjacency, weights,
                    n_iters=N_ITERS,
                    lam=lam
                )

                pred_map = np.zeros_like(segments, dtype=np.int32)
                sp_ids_arr = ft.get("sp_ids", np.unique(segments))
                sp_ids_arr = np.asarray(sp_ids_arr)

                for i, sp_id in enumerate(sp_ids_arr):
                    if i < len(preds):
                        pred_map[segments == sp_id] = preds[i] + 1

                valid = label_map > 0
                correct += int((pred_map[valid] == label_map[valid]).sum())
                total += int(valid.sum())

            acc = correct / max(total, 1)
            print(f"Accuracy: {acc:.4f}")

            results.append({
                "lambda": float(lam),
                "beta": float(beta),
                "accuracy": float(acc)
            })

    return results


def hyperparameter_grid_search_ctp(
    ds,
    test_indices,
    label_clf,
    homog_clf,
    lambda_values,
    beta_color_values,
    beta_texture_values,
    beta_pos_values,
    max_images=MAX_IMAGES
):
    results = []

    for lam in lambda_values:
        for bc in beta_color_values:
            for bt in beta_texture_values:
                for bp in beta_pos_values:
                    print(f"\nλ={lam:.3f}, βc={bc:.3f}, βt={bt:.3f}, βp={bp:.3f}")

                    correct = 0
                    total = 0

                    for idx in test_indices[:max_images]:
                        _, label_map, meta = ds[idx]
                        ft = load_ft(meta["imname"])
                        sp = load_sp(meta["imname"])

                        if ft is None or sp is None:
                            continue

                        hd = load_hypothesis(meta["imname"])
                        if hd is None:
                            continue

                        feats = ft["features"]
                        segments = sp["segments"]

                        probas = predict_image_multiH_proba(hd, label_clf, homog_clf)
                        unary = -np.log(probas + 1e-6)

                        adj_matrix = build_adjacency(segments)
                        adjacency = to_adj_dict(adj_matrix)

                        weights = compute_pairwise_weights_color_texture_position(
                            adjacency,
                            feats,
                            beta_color=bc,
                            beta_texture=bt,
                            beta_pos=bp
                        )

                        preds = belief_propagation_reparametrization(
                            unary, adjacency, weights,
                            n_iters=N_ITERS, lam=lam
                        )

                        pred_map = np.zeros_like(segments, dtype=np.int32)
                        sp_ids_arr = ft.get("sp_ids", np.unique(segments))
                        sp_ids_arr = np.asarray(sp_ids_arr)

                        for i, sp_id in enumerate(sp_ids_arr):
                            if i < len(preds):
                                pred_map[segments == sp_id] = preds[i] + 1

                        valid = label_map > 0
                        correct += int((pred_map[valid] == label_map[valid]).sum())
                        total += int(valid.sum())

                    acc = correct / max(total, 1)
                    print(f"Accuracy: {acc:.4f}")

                    results.append({
                        "lambda": float(lam),
                        "beta_color": float(bc),
                        "beta_texture": float(bt),
                        "beta_pos": float(bp),
                        "accuracy": float(acc)
                    })

    return results


def plot_results_heatmap(results, lambda_values, beta_values, save_path):
    grid = np.zeros((len(lambda_values), len(beta_values)))

    for r in results:
        i = lambda_values.index(r["lambda"])
        j = beta_values.index(r["beta"])
        grid[i, j] = r["accuracy"]

    plt.figure(figsize=(6, 5))
    plt.imshow(grid, origin="lower")

    plt.xticks(range(len(beta_values)), [f"{b:.2f}" for b in beta_values])
    plt.yticks(range(len(lambda_values)), [f"{l:.2f}" for l in lambda_values])

    plt.xlabel("beta")
    plt.ylabel("lambda")
    plt.title("Accuracy heatmap")

    for i in range(len(lambda_values)):
        for j in range(len(beta_values)):
            plt.text(
                j, i, f"{grid[i, j]:.2f}",
                ha="center", va="center",
                color="white" if grid[i, j] > 0.5 else "black"
            )

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Heatmap saved to {save_path}")

class ManualOneVsRest:
    """Same classe than 04b_multiH_classify.py, necessary to un pickle the model."""
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.estimators_ = []

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        self.estimators_ = [clone(self.base_estimator) for _ in self.classes_]

        for i, c in enumerate(self.classes_):
            y_bin = (y == c).astype(int)
            self.estimators_[i].fit(X, y_bin, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        probas = np.zeros((len(X), len(self.classes_)))
        for i, _ in enumerate(self.classes_):
            probas[:, i] = self.estimators_[i].predict_proba(X)[:, 1]

        probas /= np.maximum(probas.sum(axis=1, keepdims=True), 1e-8)
        return probas

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]
# MAIN
def main():
    ds = HoiemDataset(root_dir=DATASET_DIR)
    train_ds, test_ds = ds.get_split()

    test_indices = test_ds._i

    print("Loading MultiH classifiers...")
    with open(MODEL_DIR / "multiH_classifiers.pkl", "rb") as f:
        data = pickle.load(f)

    label_clf = data["label_clf"]
    homog_clf = data["homog_clf"]

    print("\nRunning BP inference with MultiH unary potentials...")
    acc = pixel_accuracy_bp(ds, test_indices, label_clf, homog_clf)
    print(f"Pixel accuracy (BP + MultiH init): {acc:.4f}")

    print("\nGenerating visualizations...")

    plot_confusion_bp(
        ds, test_indices, label_clf, homog_clf,
        OUT_DIR / "step6_bp_confusion_multiH.png"
    )

    plot_predictions_bp(
        ds, test_indices, label_clf, homog_clf,
        OUT_DIR / "step6_bp_predictions_multiH.png"
    )

    plot_energy_curves(
        ds, test_indices, label_clf, homog_clf,
        OUT_DIR / "step6_bp_energy_multiH.png"
    )

    with open(MODEL_DIR / "bp_model_multiH.pkl", "wb") as f:
        pickle.dump({
            "label_clf": label_clf,
            "homog_clf": homog_clf
        }, f)

    print("\ndone → outputs/step6_bp_*_multiH.png  data/models/bp_model_multiH.pkl")


def grid_search():
    ds = HoiemDataset(root_dir=DATASET_DIR)
    train_ds, test_ds = ds.get_split()
    test_indices = test_ds._i

    print("Loading MultiH classifiers...")
    with open(MODEL_DIR / "multiH_classifiers.pkl", "rb") as f:
        data = pickle.load(f)

    label_clf = data["label_clf"]
    homog_clf = data["homog_clf"]

    # Variante simple
    lambda_values = [0.05, 0.1, 0.2, 0.4]
    beta_values = [0.01, 0.02, 0.05, 0.1]

    results = hyperparameter_grid_search(
        ds, test_indices,
        label_clf, homog_clf,
        lambda_values, beta_values,
        max_images=50
    )

    save_results(results, OUT_DIR)
    plot_results_heatmap(
        results,
        lambda_values,
        beta_values,
        OUT_DIR / "grid_search_heatmap_multiH.png"
    )

    best = max(results, key=lambda x: x["accuracy"])
    print("\n=== BEST PARAMETERS ===")
    print(f"lambda = {best['lambda']}")
    print(f"beta   = {best['beta']}")
    print(f"acc    = {best['accuracy']:.4f}")


if __name__ == "__main__":
    main()
    # grid_search()