import sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pickle

sys.path.insert(0, "src")
from dataset import HoiemDataset, LABEL_NAMES, LABEL_COLORS, LABEL_IDS
from superpixels import load_sp, load_ft, build_adjacency

DATASET_DIR = Path("dataset")
MODEL_DIR   = Path("data/models"); MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR     = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)

LAMBDA_PAIRWISE = 0.5
N_ITERS = 10


# =========================
# ENERGY
# =========================
def compute_energy(labels, unary, adjacency, lam):
    E = np.sum(unary[np.arange(len(labels)), labels])

    for i in range(len(labels)):
        for j in adjacency[i]:
            if i < j and labels[i] != labels[j]:
                E += lam
    return E


# =========================
# BELIEF PROPAGATION
# =========================
def belief_propagation(unary, adjacency, n_iters=10, lam=0.5, track_energy=False):
    N, L = unary.shape
    messages = {}

    for i in range(N):
        for j in adjacency[i]:
            messages[(i, j)] = np.zeros(L)
            messages[(j, i)] = np.zeros(L)

    energy_curve = []

    for _ in range(n_iters):
        new_messages = {}

        for i in range(N):
            for j in adjacency[i]:

                msg = unary[i].copy()
                for k in adjacency[i]:
                    if k != j:
                        msg += messages[(k, i)]

                new_msg = np.zeros(L)
                for lj in range(L):
                    new_msg[lj] = min(
                        msg[li] + (0 if li == lj else lam)
                        for li in range(L)
                    )

                new_msg -= new_msg.min()
                new_messages[(i, j)] = new_msg

        messages = new_messages

        if track_energy:
            beliefs = compute_beliefs(unary, adjacency, messages)
            labels = np.argmin(beliefs, axis=1)
            energy_curve.append(compute_energy(labels, unary, adjacency, lam))

    beliefs = compute_beliefs(unary, adjacency, messages)
    labels = np.argmin(beliefs, axis=1)

    if track_energy:
        return labels, energy_curve
    return labels


def compute_beliefs(unary, adjacency, messages):
    N, L = unary.shape
    beliefs = np.zeros((N, L))

    for i in range(N):
        beliefs[i] = unary[i]
        for k in adjacency[i]:
            beliefs[i] += messages[(k, i)]

    return beliefs


# =========================
# DATA LOADING
# =========================
def load_split_data(ds, indices):
    X, y = [], []

    for idx in indices:
        _, _, m = ds[idx]
        ft = load_ft(m["imname"])
        if ft is None:
            continue

        feats = ft["features"]
        labels = ft["sp_labels"]

        mask = labels > 0
        X.append(feats[mask])
        y.append(labels[mask] - 1)

    return np.vstack(X), np.concatenate(y)


# =========================
# INFERENCE PER IMAGE
# =========================
def run_bp_on_image(ft, sp, clf, scaler):
    feats = ft["features"]
    segments = sp["segments"]

    X = scaler.transform(feats)
    probas = clf.predict_proba(X)
    unary = -np.log(probas + 1e-6)

    adjacency = build_adjacency(segments)

    preds = belief_propagation(unary, adjacency, N_ITERS, LAMBDA_PAIRWISE)

    # ⚠️ SAFE mapping
    pred_map = np.zeros_like(segments)
    sp_ids = np.unique(segments)

    for i, sp_id in enumerate(sp_ids):
        if sp_id < len(preds):
            pred_map[segments == sp_id] = preds[sp_id] + 1

    return pred_map


# =========================
# METRICS
# =========================
def pixel_accuracy_bp(ds, indices, clf, scaler):
    correct = total = 0

    for idx in indices:
        _, label_map, m = ds[idx]
        ft = load_ft(m["imname"])
        sp = load_sp(m["imname"])
        if ft is None or sp is None:
            continue

        pred_map = run_bp_on_image(ft, sp, clf, scaler)

        valid = label_map > 0
        correct += (pred_map[valid] == label_map[valid]).sum()
        total += valid.sum()

    return correct / total


# =========================
# PLOTS
# =========================
def plot_confusion_bp(ds, indices, clf, scaler, save_path):
    all_preds, all_true = [], []

    for idx in indices[:50]:
        _, label_map, m = ds[idx]
        ft = load_ft(m["imname"])
        sp = load_sp(m["imname"])
        if ft is None or sp is None:
            continue

        pred_map = run_bp_on_image(ft, sp, clf, scaler)

        valid = label_map > 0
        all_true.extend(label_map[valid].tolist())
        all_preds.extend(pred_map[valid].tolist())

    cm = confusion_matrix(all_true, all_preds, labels=LABEL_IDS, normalize="true")

    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    plt.colorbar(im)

    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels([LABEL_NAMES[l] for l in LABEL_IDS])
    ax.set_yticklabels([LABEL_NAMES[l] for l in LABEL_IDS])

    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center")

    plt.title("Confusion Matrix (BP)")
    plt.savefig(save_path)
    plt.close()


def plot_predictions_bp(ds, indices, clf, scaler, save_path, n=6):
    fig, axes = plt.subplots(3, n, figsize=(4*n, 12))

    for col, idx in enumerate(indices[:n]):
        image, label_map, m = ds[idx]
        ft = load_ft(m["imname"])
        sp = load_sp(m["imname"])
        if ft is None or sp is None:
            continue

        pred_map = run_bp_on_image(ft, sp, clf, scaler)

        gt_rgb = np.zeros((*label_map.shape, 3), dtype=np.uint8)
        pred_rgb = np.zeros((*label_map.shape, 3), dtype=np.uint8)

        for l, c in LABEL_COLORS.items():
            gt_rgb[label_map == l] = c
            pred_rgb[pred_map == l] = c

        axes[0, col].imshow(image)
        axes[1, col].imshow(gt_rgb)
        axes[2, col].imshow(pred_rgb)

        for r in range(3):
            axes[r, col].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_energy_curve(curve, save_path):
    plt.figure(figsize=(6,4))
    plt.plot(curve, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title("BP convergence")
    plt.grid()
    plt.savefig(save_path)
    plt.close()


# =========================
# MAIN
# =========================
def main():
    ds = HoiemDataset(root_dir=DATASET_DIR)
    train_ds, test_ds = ds.get_split()

    train_indices = train_ds._i
    test_indices = test_ds._i

    print("Loading features...")
    X_tr, y_tr = load_split_data(ds, train_indices)

    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)

    clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=200,
        random_state=42,
    )
    clf.fit(X_tr, y_tr)

    print("\nRunning BP inference...")
    acc = pixel_accuracy_bp(ds, test_indices, clf, scaler)
    print(f"Pixel accuracy (BP): {acc:.4f}")

    plot_predictions_bp(ds, test_indices, clf, scaler,
                        OUT_DIR / "step6_bp_predictions.png")

    plot_confusion_bp(ds, test_indices, clf, scaler,
                      OUT_DIR / "step6_confusion.png")

    # convergence on one image
    _, _, m0 = ds[test_indices[0]]
    ft0 = load_ft(m0["imname"])
    sp0 = load_sp(m0["imname"])

    X0 = scaler.transform(ft0["features"])
    unary0 = -np.log(clf.predict_proba(X0) + 1e-6)
    adj0 = build_adjacency(sp0["segments"])

    _, energy_curve = belief_propagation(
        unary0, adj0, N_ITERS, LAMBDA_PAIRWISE, track_energy=True
    )

    plot_energy_curve(energy_curve, OUT_DIR / "step6_energy.png")

    with open(MODEL_DIR / "bp_model.pkl", "wb") as f:
        pickle.dump({"clf": clf, "scaler": scaler}, f)


if __name__ == "__main__":
    main()