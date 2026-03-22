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
OUT_DIR     = Path("outputs"); OUT_DIR.mkdir(parents=True, exist_ok=True)

LAMBDA_PAIRWISE = 0.15
N_ITERS = 10

# ENERGY
def compute_energy(labels, unary, adjacency, lam, weights):
    E = np.sum(unary[np.arange(len(labels)), labels])

    for i in range(len(labels)):
        for j in adjacency[i]:
            if i < j and labels[i] != labels[j]:
                w = weights.get((i, j), 1.0)
                E += lam * w

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

# NEW: SUPERPIXEL COLORS
def compute_superpixel_colors(image, segments):
    sp_ids = np.unique(segments)
    colors = np.zeros((len(sp_ids), 3))

    for i, sp_id in enumerate(sp_ids):
        mask = segments == sp_id
        colors[i] = image[mask].mean(axis=0)

    return colors

# PAIRWISE WEIGHTS (COLOR-BASED)
def compute_pairwise_weights(adjacency, features, beta=0.05):
    weights = {}

    for i in adjacency:
        for j in adjacency[i]:
            diff = features[i] - features[j]
            d = np.dot(diff, diff)
            w = np.exp(-beta * d)

            weights[(i, j)] = w
            weights[(j, i)] = w

    return weights


# BELIEF PROPAGATION
def belief_propagation_reparametrization(unary, adjacency, weights, n_iters=20, lam=0.5,
                      damping=0.0, track_energy=False):

    N, L = unary.shape

    adj = to_adj_dict(adjacency)
    for i in list(adj.keys()):
        for j in adj[i]:
            adj.setdefault(j, set()).add(i)
    adj = {i: list(v) for i, v in adj.items()}

    theta = unary.copy()

    energy_curve = []

    for _ in range(n_iters):

        for i in range(N):
            for j in adj[i]:

                w = weights.get((i, j), 1.0)

                # COMPUTE MESSAGE (M_ij)
                M = np.zeros(L)

                for lj in range(L):

                    vals = []

                    for li in range(L):

                        # Potts pairwise
                        if li == lj:
                            pairwise = 0
                        else:
                            pairwise = lam * w

                        vals.append(theta[i, li] + pairwise)

                    M[lj] = min(vals)

                # REPARAMETRIZATION
                # theta_j += M_ij
                theta[j] += M

                # normalisation
                theta[j] -= theta[j].min()

        # ENERGY TRACKING
        if track_energy:

            labels_tmp = np.argmin(theta, axis=1)

            adj_list = {i: adj[i] for i in range(N)}
            energy = compute_energy(labels_tmp, unary, adj_list, lam, weights)

            energy_curve.append(energy)

    # FINAL LABELS
    labels = np.argmin(theta, axis=1)

    if track_energy:
        return labels, energy_curve
    return labels

def belief_propagation_messages(unary, adjacency, weights, n_iters=20, lam=0.5,
                      damping=0.5, track_energy=False):

    N, L = unary.shape

    adj = to_adj_dict(adjacency)
    for i in list(adj.keys()):
        for j in adj[i]:
            adj.setdefault(j, set()).add(i)
    adj = {i: list(v) for i, v in adj.items()}

    messages = {(i, j): np.zeros(L) for i in adj for j in adj[i]}

    energy_curve = []

    for _ in range(n_iters):

        new_messages = {}

        for i in range(N):
            for j in adj[i]:

                # messages
                msg = unary[i].copy()
                for k in adj[i]:
                    if k != j:
                        msg += messages[(k, i)]

                w = weights.get((i, j), 1.0)

                # BP Potts
                min_msg = msg.min()

                new_msg = np.zeros(L)
                for l in range(L):
                    new_msg[l] = min(
                        msg[l],
                        min_msg + lam * w
                    )

                # normalisation
                new_msg -= new_msg.min()

                # damping
                new_msg = damping * new_msg + (1 - damping) * messages[(i, j)]

                new_messages[(i, j)] = new_msg

        messages = new_messages

        if track_energy:
            beliefs_tmp = np.zeros((N, L))
            for i in range(N):
                beliefs_tmp[i] = unary[i]
                for k in adj[i]:
                    beliefs_tmp[i] += messages[(k, i)]

            labels_tmp = np.argmin(beliefs_tmp, axis=1)

            adj_list = {i: adj[i] for i in range(N)}
            energy = compute_energy(labels_tmp, unary, adj_list, lam, weights)
            energy_curve.append(energy)

    beliefs = np.zeros((N, L))
    for i in range(N):
        beliefs[i] = unary[i]
        for k in adj[i]:
            beliefs[i] += messages[(k, i)]

    labels = np.argmin(beliefs, axis=1)

    if track_energy:
        return labels, energy_curve
    return labels


# DATA LOADING
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


# INFERENCE PER IMAGE
def run_bp_on_image(ft, sp, clf, scaler, track_energy=False, reparametrization=True):
    feats = ft["features"]
    segments = sp["segments"]
    
    X = scaler.transform(feats)
    probas = clf.predict_proba(X)
    unary = -np.log(probas + 1e-6)

    adj_matrix = build_adjacency(segments)
    adjacency = to_adj_dict(adj_matrix)

    weights = compute_pairwise_weights(adjacency, X, beta=0.05)
    if track_energy:
        if reparametrization:
            preds, energy_curve = belief_propagation_reparametrization(
                unary, adjacency, weights,
                N_ITERS, LAMBDA_PAIRWISE,
                track_energy=True
            )
        else:
            preds, energy_curve = belief_propagation_messages(
                unary, adjacency, weights,
                N_ITERS, LAMBDA_PAIRWISE,
                track_energy=True
            )
    else:
        if reparametrization:
            preds = belief_propagation_reparametrization(
                unary, adjacency, weights,
                N_ITERS, LAMBDA_PAIRWISE
            )
        else:
            preds = belief_propagation_messages(
                unary, adjacency, weights,
                N_ITERS, LAMBDA_PAIRWISE
            )

    pred_map = np.zeros_like(segments)
    sp_ids = np.unique(segments)
    sp_ids_sorted = np.sort(sp_ids)


    for i, sp_id in enumerate(sp_ids_sorted):
        pred_map[segments == sp_id] = preds[i] + 1

    if track_energy:
        return pred_map, energy_curve
    return pred_map


# METRICS
def pixel_accuracy_bp(ds, indices, clf, scaler, reparametrization=True):
    correct = total = 0

    for idx in indices:
        _, label_map, m = ds[idx]
        ft = load_ft(m["imname"])
        sp = load_sp(m["imname"])
        if ft is None or sp is None:
            continue

        pred_map = run_bp_on_image(ft, sp, clf, scaler, reparametrization=reparametrization)

        valid = label_map > 0
        correct += (pred_map[valid] == label_map[valid]).sum()
        total += valid.sum()

    return correct / total


# PLOTS
def plot_confusion_bp(ds, indices, clf, scaler, save_path, reparametrization=True):
    all_preds, all_true = [], []

    for idx in indices[:50]:
        _, label_map, m = ds[idx]
        ft = load_ft(m["imname"])
        sp = load_sp(m["imname"])
        if ft is None or sp is None:
            continue

        pred_map = run_bp_on_image(ft, sp, clf, scaler, reparametrization=reparametrization)

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
    ax.set_title("BP confusion matrix")

    for i in range(len(LABEL_IDS)):
        for j in range(len(LABEL_IDS)):
            ax.text(j, i, f"{cm[i,j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm[i,j] > 0.5 else "black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_predictions_bp(ds, indices, clf, scaler, save_path, n=6, reparametrization=True):
    fig, axes = plt.subplots(3, n, figsize=(4*n, 12))
    fig.suptitle("BP predictions", fontsize=13)

    for col, idx in enumerate(indices[:n]):
        image, label_map, m = ds[idx]
        ft = load_ft(m["imname"])
        sp = load_sp(m["imname"])
        if ft is None or sp is None:
            continue

        pred_map = run_bp_on_image(ft, sp, clf, scaler, reparametrization=reparametrization)

        valid = label_map > 0
        acc = (pred_map[valid] == label_map[valid]).mean()

        gt_rgb   = np.zeros((*label_map.shape, 3), dtype=np.uint8)
        pred_rgb = np.zeros((*label_map.shape, 3), dtype=np.uint8)

        for l, c in LABEL_COLORS.items():
            gt_rgb[label_map == l]   = c
            pred_rgb[pred_map == l]  = c

        axes[0, col].imshow(image)
        axes[0, col].set_title(Path(m["imname"]).stem, fontsize=7)

        axes[1, col].imshow(gt_rgb)
        axes[1, col].set_title("GT", fontsize=7)

        axes[2, col].imshow(pred_rgb)
        axes[2, col].set_title(f"BP acc={acc:.2f}", fontsize=7)

        for r in range(3):
            axes[r, col].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def plot_energy_curves(ds, indices, clf, scaler, save_path, n=5, reparametrization=True):
    fig, ax = plt.subplots(figsize=(6, 4))

    for idx in indices[:n]:
        _, _, m = ds[idx]
        ft = load_ft(m["imname"])
        sp = load_sp(m["imname"])
        if ft is None or sp is None:
            continue

        _, energy_curve = run_bp_on_image(
            ft, sp, clf, scaler, track_energy=True, reparametrization=True
        )

        ax.plot(energy_curve, marker="o", markersize=2,
                label=Path(m["imname"]).stem)

    ax.set_xlabel("iteration")
    ax.set_ylabel("energy")
    ax.set_title("BP energy convergence")
    ax.legend(fontsize=6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# MAIN
def main(reparametrization=True):
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

    print("\nGenerating visualizations...")

    plot_confusion_bp(
        ds, test_indices, clf, scaler,
        OUT_DIR / "step6_bp_confusion.png"
    )

    plot_predictions_bp(
        ds, test_indices, clf, scaler,
        OUT_DIR / "step6_bp_predictions.png"
    )

    plot_energy_curves(
        ds, test_indices, clf, scaler,
        OUT_DIR / "step6_bp_energy.png"
    )

    with open(MODEL_DIR / "bp_model.pkl", "wb") as f:
        pickle.dump({"clf": clf, "scaler": scaler}, f)


if __name__ == "__main__":
    main()