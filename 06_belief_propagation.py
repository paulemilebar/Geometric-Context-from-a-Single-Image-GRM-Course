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

# HYPERPARAMETERS SET UP
LAMBDA_PAIRWISE = 0.1
BETA = 0.05
BETA_COLOR = 0.2
BETA_TEXTURE = 0.1
BETA_POS = 0.08
N_ITERS = 15
MAX_IMAGES = 90 # fro grid search
BETA = 0.05

# ENERGY COMPUTATION
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

# PAIRWISE WEIGHTS (ALL FEATURES-BASED)
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

# PAIRWISE WEIGHTS (DIFFERENT WEIGHTS FOR DIFFERENT TYPE OF FEATURES)
def compute_pairwise_weights_color_texture_position(adjacency, features,
                             beta_color=BETA_COLOR,
                             beta_texture=BETA_TEXTURE,
                             beta_pos=BETA_POS):

    weights = {}

    color_idx   = slice(0, 16)
    texture_idx = slice(16, 31)
    geometry_idx = slice(31, 43)

    # extract position (x,y mean) 
    # L1 = mean x,y
    pos_idx = slice(31, 33)

    for i in adjacency:
        for j in adjacency[i]:

            # COLOR
            diff_color = features[i][color_idx] - features[j][color_idx]
            d_color = np.dot(diff_color, diff_color)

            # TEXTURE
            diff_texture = features[i][texture_idx] - features[j][texture_idx]
            d_texture = np.dot(diff_texture, diff_texture)

            # POSITION
            diff_pos = features[i][pos_idx] - features[j][pos_idx]
            d_pos = np.dot(diff_pos, diff_pos)

            # COMBINAISON
            w = np.exp(
                - beta_color * d_color
                - beta_texture * d_texture
                - beta_pos * d_pos
            )

            weights[(i, j)] = w
            weights[(j, i)] = w

    return weights


# BELIEF PROPAGATION 
def compute_energy_full(labels, unary, pairwise, adjacency):
    E = 0.0

    # unary
    for i in range(len(labels)):
        E += unary[i, labels[i]]

    # pairwise
    for i in adjacency:
        for j in adjacency[i]:
            if i < j:
                E += pairwise[(i, j)][labels[i], labels[j]]

    return E


def belief_propagation_reparametrization(unary_init, adjacency, weights,
                                         n_iters=20, lam=0.5,
                                         track_energy=False):

    N, L = unary_init.shape

    # adjacency
    adj = to_adj_dict(adjacency)
    for i in list(adj.keys()):
        for j in adj[i]:
            adj.setdefault(j, set()).add(i)
    adj = {i: list(v) for i, v in adj.items()}

    # unic edges
    edges = []
    for i in adj:
        for j in adj[i]:
            if i < j:
                edges.append((i, j))

    # initialization
    unary = unary_init.copy()

    pairwise = {}
    for (i, j) in edges:
        w = weights.get((i, j), 1.0)

        V = np.full((L, L), lam * w)
        np.fill_diagonal(V, 0)

        pairwise[(i, j)] = V.copy()
        pairwise[(j, i)] = V.T.copy()

    energy_curve = []

    # BP iterations
    for _ in range(n_iters):

        # FORWARD PASS
        for (i, j) in edges:

            V = pairwise[(i, j)]

            M = np.zeros(L)
            for lj in range(L):
                M[lj] = np.min(unary[i] + V[:, lj])

            # reparametrization
            unary[j] += M
            pairwise[(i, j)] -= M[None, :]
            pairwise[(j, i)] = pairwise[(i, j)].T

            # normalisation
            unary[j] -= unary[j].min()

        # BACKWARD PASS
        for (i, j) in reversed(edges):

            V = pairwise[(j, i)]

            M = np.zeros(L)
            for li in range(L):
                M[li] = np.min(unary[j] + V[:, li])

            unary[i] += M
            pairwise[(j, i)] -= M[None, :]
            pairwise[(i, j)] = pairwise[(j, i)].T

            unary[i] -= unary[i].min()

        # -------- energy tracking --------
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
def run_bp_on_image(ft, sp, clf, scaler, track_energy=False):
    feats = ft["features"]
    segments = sp["segments"]
    
    X = scaler.transform(feats)
    probas = clf.predict_proba(X)
    unary = -np.log(probas + 1e-6)

    adj_matrix = build_adjacency(segments)
    adjacency = to_adj_dict(adj_matrix)

    # 2 DIFFERENT APPROACHES : 
    # either we separate the the feature comparaision based on colors, location and texture. compute_pairwise_weights_color_texture_position()
    # Or we just compute the comparaision based on ALL features. compute_pairwise_weights()

    weights = compute_pairwise_weights_color_texture_position(adjacency, X, beta_color=BETA_COLOR, beta_texture=BETA_TEXTURE, beta_pos=BETA_POS)
    #weights = compute_pairwise_weights(adjacency, X, beta=BETA)
    if track_energy:
        preds, energy_curve = belief_propagation_reparametrization(
                unary, adjacency, weights,
                N_ITERS, LAMBDA_PAIRWISE,
                track_energy=True
            )
    else:
        preds = belief_propagation_reparametrization(
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


# PLOTS
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


def plot_predictions_bp(ds, indices, clf, scaler, save_path, n=6):
    fig, axes = plt.subplots(3, n, figsize=(4*n, 12))
    fig.suptitle("BP predictions", fontsize=13)

    for col, idx in enumerate(indices[:n]):
        image, label_map, m = ds[idx]
        ft = load_ft(m["imname"])
        sp = load_sp(m["imname"])
        if ft is None or sp is None:
            continue

        pred_map = run_bp_on_image(ft, sp, clf, scaler)

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


def plot_energy_curves(ds, indices, clf, scaler, save_path, n=5):
    fig, ax = plt.subplots(figsize=(6, 4))

    for idx in indices[:n]:
        _, _, m = ds[idx]
        ft = load_ft(m["imname"])
        sp = load_sp(m["imname"])
        if ft is None or sp is None:
            continue

        _, energy_curve = run_bp_on_image(
            ft, sp, clf, scaler, track_energy=True
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



import json
import csv
from datetime import datetime


def save_results(results, out_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = out_dir / f"grid_search_{timestamp}.json"
    csv_path  = out_dir / f"grid_search_{timestamp}.csv"

    # JSON
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    # CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lambda", "beta", "accuracy"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\nResults saved to:\n{json_path}\n{csv_path}")


def hyperparameter_grid_search(ds, test_indices, clf, scaler,
                               lambda_values, beta_values,
                               max_images=MAX_IMAGES
                               ):

    results = []

    for lam in lambda_values:
        for beta in beta_values:

            print(f"\nTesting lambda={lam:.3f}, beta={beta:.3f}")

            correct = 0
            total = 0

            for idx in test_indices[:max_images]:

                _, label_map, m = ds[idx]
                ft = load_ft(m["imname"])
                sp = load_sp(m["imname"])

                if ft is None or sp is None:
                    continue

                feats = ft["features"]
                segments = sp["segments"]

                X = scaler.transform(feats)
                probas = clf.predict_proba(X)
                unary = -np.log(probas + 1e-6)

                adj_matrix = build_adjacency(segments)
                adjacency = to_adj_dict(adj_matrix)

                weights = compute_pairwise_weights(adjacency, X, beta=beta)

                preds = belief_propagation_reparametrization(
                        unary, adjacency, weights,
                        n_iters=N_ITERS, lam=lam
                    )
                

                pred_map = np.zeros_like(segments)
                sp_ids = np.unique(segments)

                for i, sp_id in enumerate(np.sort(sp_ids)):
                    pred_map[segments == sp_id] = preds[i] + 1

                valid = label_map > 0
                correct += (pred_map[valid] == label_map[valid]).sum()
                total += valid.sum()

            acc = correct / total if total > 0 else 0
            print(f"Accuracy: {acc:.4f}")

            results.append({
                "lambda": float(lam),
                "beta": float(beta),
                "accuracy": float(acc)
            })

    return results


def plot_results_heatmap(results, lambda_values, beta_values, save_path):

    grid = np.zeros((len(lambda_values), len(beta_values)))

    for r in results:
        i = lambda_values.index(r["lambda"])
        j = beta_values.index(r["beta"])
        grid[i, j] = r["accuracy"]

    plt.figure(figsize=(6,5))
    plt.imshow(grid, origin="lower")

    plt.xticks(range(len(beta_values)), [f"{b:.2f}" for b in beta_values])
    plt.yticks(range(len(lambda_values)), [f"{l:.2f}" for l in lambda_values])

    plt.xlabel("beta")
    plt.ylabel("lambda")
    plt.title("Accuracy heatmap")

    for i in range(len(lambda_values)):
        for j in range(len(beta_values)):
            plt.text(j, i, f"{grid[i,j]:.2f}",
                     ha="center", va="center",
                     color="white" if grid[i,j] > 0.5 else "black")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Heatmap saved to {save_path}")

def hyperparameter_grid_search_ctp(
    ds, test_indices, clf, scaler,
    lambda_values,
    beta_color_values,
    beta_texture_values,
    beta_pos_values,
    max_images=MAX_IMAGES,
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

                        _, label_map, m = ds[idx]
                        ft = load_ft(m["imname"])
                        sp = load_sp(m["imname"])

                        if ft is None or sp is None:
                            continue

                        feats = ft["features"]
                        segments = sp["segments"]

                        X = scaler.transform(feats)
                        probas = clf.predict_proba(X)
                        unary = -np.log(probas + 1e-6)

                        adj_matrix = build_adjacency(segments)
                        adjacency = to_adj_dict(adj_matrix)

                        weights = compute_pairwise_weights_color_texture_position(
                            adjacency, X,
                            beta_color=bc,
                            beta_texture=bt,
                            beta_pos=bp
                        )

                        preds = belief_propagation_reparametrization(
                                unary, adjacency, weights,
                                n_iters=N_ITERS, lam=lam
                            )
                        

                        pred_map = np.zeros_like(segments)
                        sp_ids = np.unique(segments)

                        for i, sp_id in enumerate(np.sort(sp_ids)):
                            pred_map[segments == sp_id] = preds[i] + 1

                        valid = label_map > 0
                        correct += (pred_map[valid] == label_map[valid]).sum()
                        total += valid.sum()

                    acc = correct / total if total > 0 else 0
                    print(f"Accuracy: {acc:.4f}")

                    results.append({
                        "lambda": float(lam),
                        "beta_color": float(bc),
                        "beta_texture": float(bt),
                        "beta_pos": float(bp),
                        "accuracy": float(acc)
                    })

    return results


def grid_search():

    print("=== DATA LOADING ===")
    ds = HoiemDataset(root_dir=DATASET_DIR)
    train_ds, test_ds = ds.get_split()

    train_indices = train_ds._i
    test_indices = test_ds._i

    print("Loading features...")
    X_tr, y_tr = load_split_data(ds, train_indices)

    scaler = StandardScaler().fit(X_tr)
    X_tr = scaler.transform(X_tr)

    print("Training AdaBoost...")
    clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=200,
        random_state=42,
    )
    clf.fit(X_tr, y_tr)

    # =========================
    # GRID SEARCH PARAMETERS WITOUT SEPARATED FEATURES
    # =========================

    #lambda_values = [0.05, 0.1, 0.2, 0.4, 0.8]
    #beta_values   = [0.01, 0.02, 0.05, 0.1]

    #print("\n=== START GRID SEARCH ===")

    #results = hyperparameter_grid_search(
        #ds, test_indices, clf, scaler,
        #lambda_values, beta_values,
        #max_images=100, 
    #)

    # =========================
    # GRID SEARCH PARAMETERS WITH SEPARATED FEATURES COLORS POSITION AND TEXTURE
    # =========================

    lambda_values = [0.1, 0.2]

    beta_color_values   = [0.03, 0.05, 0.1, 0.2]
    beta_texture_values = [0.03, 0.05, 0.1]
    beta_pos_values     = [0.01, 0.02, 0.05, 0.08]

    results = hyperparameter_grid_search_ctp(
    ds, test_indices, clf, scaler,
    lambda_values,
    beta_color_values,
    beta_texture_values,
    beta_pos_values,
    max_images=50
    )

    # =========================
    # SAVE RESULTS
    # =========================

    save_results(results, OUT_DIR)

    # =========================
    # HEATMAP
    # =========================

    #plot_results_heatmap(
    #    results,
    #    lambda_values,
    #    beta_values,
    #    OUT_DIR / "grid_search_heatmap.png"
   # )

    # BEST PARAMS

    best = max(results, key=lambda x: x["accuracy"])

    print("\n=== BEST PARAMETERS ===")
    print(f"lambda = {best['lambda']}")
    print(f"beta   = {best['beta']}")
    print(f"acc    = {best['accuracy']:.4f}")

## NOTE : IF YOU WANT TO DO A GRID SEARCH TO STUDY THE INFLUENCE ON HYPERPARAMETERS LAMBDA AND BETA UNCOMMENT grid_search() AND COMMENT main(). 
if __name__ == "__main__":
    main()
    #grid_search()