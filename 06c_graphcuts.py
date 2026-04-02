import sys
import json
import csv
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

sys.path.insert(0, "src")
from dataset import HoiemDataset, LABEL_NAMES, LABEL_COLORS, LABEL_IDS
from superpixels import load_sp, load_ft, build_adjacency


DATASET_DIR = Path("dataset")
MODEL_DIR   = Path("data/models"); MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR     = Path("outputs"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# HYPERPARAMETERS
LAMBDA_PAIRWISE = 0.1
BETA = 0.05
BETA_COLOR = 0.2
BETA_TEXTURE = 0.1
BETA_POS = 0.05
MAX_IMAGES = 90

# alpha-expansion
MAX_EXPANSION_ITERS = 20
GC_EPS = 1e-10


# UTILS
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


def unique_undirected_edges(adjacency):
    adj = to_adj_dict(adjacency)
    edges = set()
    for i in adj:
        for j in adj[i]:
            a, b = sorted((int(i), int(j)))
            if a != b:
                edges.add((a, b))
    return sorted(edges)


# PAIRWISE WEIGHTS
def compute_pairwise_weights(adjacency, features, beta=BETA):
    weights = {}
    for i in adjacency:
        for j in adjacency[i]:
            diff = features[i] - features[j]
            d = float(np.dot(diff, diff))
            w = float(np.exp(-beta * d))
            weights[(i, j)] = w
            weights[(j, i)] = w
    return weights


def compute_pairwise_weights_color_texture_position(
    adjacency, features,
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
            d_color = float(np.dot(diff_color, diff_color))

            diff_texture = features[i][texture_idx] - features[j][texture_idx]
            d_texture = float(np.dot(diff_texture, diff_texture))

            diff_pos = features[i][pos_idx] - features[j][pos_idx]
            d_pos = float(np.dot(diff_pos, diff_pos))

            w = float(np.exp(
                - beta_color * d_color
                - beta_texture * d_texture
                - beta_pos * d_pos
            ))

            weights[(i, j)] = w
            weights[(j, i)] = w

    return weights


# ENERGY
def compute_multilabel_energy(labels, unary, adjacency, weights, lam):
    E = float(np.sum(unary[np.arange(len(labels)), labels]))
    for i, j in unique_undirected_edges(adjacency):
        if labels[i] != labels[j]:
            E += lam * weights.get((i, j), 1.0)
    return E


# BINARY GRAPH-CUT REDUCTION
def add_unary_cost(G, node, cost0, cost1, source="s", sink="t"):
    """
    x=0 <=> node in source side
    x=1 <=> node in sink side

    To encode unary:
        cost(x=0) = cost0
        cost(x=1) = cost1

    add edge source->node with cap=cost1
    add edge node->sink with cap=cost0
    """
    if cost1 < 0 or cost0 < 0:
        raise ValueError("Unary capacities must be non-negative at graph level.")

    if cost1 > 0:
        G.add_edge(source, node, capacity=G.get_edge_data(source, node, {}).get("capacity", 0.0) + float(cost1))
    if cost0 > 0:
        G.add_edge(node, sink, capacity=G.get_edge_data(node, sink, {}).get("capacity", 0.0) + float(cost0))


def add_linear_term(G, node, coeff, constant_shift, source="s", sink="t"):
    """
    Add coeff * x, x in {0,1}, with x=1 on sink side.
    If coeff >= 0: cost(0)=0, cost(1)=coeff
    If coeff < 0: coeff*x = coeff + (-coeff)*(1-x)
    """
    if coeff >= 0:
        add_unary_cost(G, node, cost0=0.0, cost1=coeff, source=source, sink=sink)
    else:
        constant_shift[0] += coeff
        add_unary_cost(G, node, cost0=-coeff, cost1=0.0, source=source, sink=sink)


def add_submodular_pairwise(G, node_i, node_j, A, B, C, D, constant_shift, source="s", sink="t"):
    """
    Encode binary pairwise term f(x_i, x_j) with table:
        A = f(0,0)
        B = f(0,1)
        C = f(1,0)
        D = f(1,1)

    Requires submodularity:
        B + C >= A + D

    Decomposition:
        f(x_i, x_j) = A
                     + (C - A) x_i
                     + (D - C) x_j
                     + (B + C - A - D) (1 - x_i) x_j
    """
    if B + C < A + D - 1e-12:
        raise ValueError(
            f"Non-submodular pairwise term: A={A}, B={B}, C={C}, D={D}"
        )

    constant_shift[0] += A

    add_linear_term(G, node_i, C - A, constant_shift, source=source, sink=sink)
    add_linear_term(G, node_j, D - C, constant_shift, source=source, sink=sink)

    k = B + C - A - D
    if k > 0:
        G.add_edge(node_i, node_j, capacity=G.get_edge_data(node_i, node_j, {}).get("capacity", 0.0) + float(k))


# ALPHA-EXPANSION MOVE
def alpha_expansion_move(current_labels, alpha, unary, adjacency, weights, lam):
    """
    One alpha-expansion move for weighted Potts model:
        E(y)=sum_i D_i(y_i)+sum_{ij} lam*w_ij*[y_i != y_j]

    current_labels: shape (N,), label indices in {0,...,L-1}
    alpha: candidate label
    unary: shape (N,L)
    """
    N, L = unary.shape
    source, sink = "s", "t"

    # Active nodes: those not already alpha
    active_nodes = [i for i in range(N) if current_labels[i] != alpha]
    if not active_nodes:
        return current_labels.copy(), compute_multilabel_energy(current_labels, unary, adjacency, weights, lam)

    active_set = set(active_nodes)

    G = nx.DiGraph()
    G.add_node(source)
    G.add_node(sink)
    for i in active_nodes:
        G.add_node(i)

    constant_shift = [0.0]

    # Unary terms:
    # x_i = 0 => keep current label
    # x_i = 1 => switch to alpha
    for i in active_nodes:
        keep_cost = float(unary[i, current_labels[i]])
        switch_cost = float(unary[i, alpha])
        add_unary_cost(G, i, cost0=keep_cost, cost1=switch_cost, source=source, sink=sink)

    # Pairwise terms
    for i, j in unique_undirected_edges(adjacency):
        w = float(lam * weights.get((i, j), 1.0))
        if w <= 0:
            continue

        i_active = i in active_set
        j_active = j in active_set

        if i_active and j_active:
            li = current_labels[i]
            lj = current_labels[j]

            # Binary table for expansion variables x_i, x_j:
            # x=0 keep current, x=1 switch to alpha
            # resulting labels:
            #   label_i(x_i) = li if x_i=0 else alpha
            #   label_j(x_j) = lj if x_j=0 else alpha
            A = w if li != lj else 0.0  # f(0,0)
            B = w                       # f(0,1), li != alpha
            C = w                       # f(1,0), lj != alpha
            D = 0.0                     # f(1,1)

            add_submodular_pairwise(G, i, j, A, B, C, D, constant_shift, source=source, sink=sink)

        elif i_active and not j_active:
            # j fixed to alpha
            # cost is w if i keeps its non-alpha label, else 0
            add_unary_cost(G, i, cost0=w, cost1=0.0, source=source, sink=sink)

        elif not i_active and j_active:
            # i fixed to alpha
            add_unary_cost(G, j, cost0=w, cost1=0.0, source=source, sink=sink)

        else:
            # both already alpha => pairwise cost always 0
            pass

    cut_value, (S, T) = nx.minimum_cut(G, source, sink, capacity="capacity", flow_func=nx.algorithms.flow.preflow_push)

    new_labels = current_labels.copy()
    for i in active_nodes:
        x_i = 0 if i in S else 1
        if x_i == 1:
            new_labels[i] = alpha

    E_new = compute_multilabel_energy(new_labels, unary, adjacency, weights, lam)
    return new_labels, E_new


def alpha_expansion(unary, adjacency, weights, lam, max_iters=MAX_EXPANSION_ITERS, track_energy=False):
    """
    Multi-label optimization by alpha-expansion.
    """
    N, L = unary.shape

    # Initialization: unary argmin
    labels = np.argmin(unary, axis=1).astype(int)
    current_energy = compute_multilabel_energy(labels, unary, adjacency, weights, lam)

    energy_curve = [current_energy] if track_energy else None

    for _ in range(max_iters):
        improved = False

        for alpha in range(L):
            candidate_labels, candidate_energy = alpha_expansion_move(
                labels, alpha, unary, adjacency, weights, lam
            )

            if candidate_energy < current_energy - GC_EPS:
                labels = candidate_labels
                current_energy = candidate_energy
                improved = True

                if track_energy:
                    energy_curve.append(current_energy)

        if not improved:
            break

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
        y.append(labels[mask])

    return np.vstack(X), np.concatenate(y)


# INFERENCE PER IMAGE
def run_gc_on_image(ft, sp, clf, scaler, track_energy=False):
    feats = ft["features"]
    segments = sp["segments"]

    X = scaler.transform(feats)
    probas = clf.predict_proba(X)
    unary = -np.log(probas + 1e-6)

    adj_matrix = build_adjacency(segments)
    adjacency = to_adj_dict(adj_matrix)

    # Choose one version
    weights = compute_pairwise_weights(adjacency, X, beta=BETA)
    # weights = compute_pairwise_weights_color_texture_position(
    #     adjacency, X,
    #     beta_color=BETA_COLOR,
    #     beta_texture=BETA_TEXTURE,
    #     beta_pos=BETA_POS
    # )

    if track_energy:
        preds, energy_curve = alpha_expansion(
            unary, adjacency, weights,
            lam=LAMBDA_PAIRWISE,
            max_iters=MAX_EXPANSION_ITERS,
            track_energy=True
        )
    else:
        preds = alpha_expansion(
            unary, adjacency, weights,
            lam=LAMBDA_PAIRWISE,
            max_iters=MAX_EXPANSION_ITERS,
            track_energy=False
        )

    pred_map = np.zeros_like(segments)
    sp_ids = np.unique(segments)
    sp_ids_sorted = np.sort(sp_ids)
    classes = clf.classes_

    # preds are label-indices in [0, L-1], classes[preds[i]] are dataset labels
    for i, sp_id in enumerate(sp_ids_sorted):
        pred_map[segments == sp_id] = classes[preds[i]]

    if track_energy:
        return pred_map, energy_curve
    return pred_map


# METRICS
def pixel_accuracy_gc(ds, indices, clf, scaler):
    correct = total = 0

    for idx in indices:
        _, label_map, m = ds[idx]
        ft = load_ft(m["imname"])
        sp = load_sp(m["imname"])
        if ft is None or sp is None:
            continue

        pred_map = run_gc_on_image(ft, sp, clf, scaler)

        valid = label_map > 0
        correct += (pred_map[valid] == label_map[valid]).sum()
        total += valid.sum()

    return correct / total if total > 0 else 0.0


# PLOTS
def plot_confusion_gc(ds, indices, clf, scaler, save_path):
    all_preds, all_true = [], []

    for idx in indices[:50]:
        _, label_map, m = ds[idx]
        ft = load_ft(m["imname"])
        sp = load_sp(m["imname"])
        if ft is None or sp is None:
            continue

        pred_map = run_gc_on_image(ft, sp, clf, scaler)

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
    ax.set_title("Graph Cut confusion matrix")

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


def plot_predictions_gc(ds, indices, clf, scaler, save_path, n=6):
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 12))
    fig.suptitle("Graph Cut predictions", fontsize=13)

    for col, idx in enumerate(indices[:n]):
        image, label_map, m = ds[idx]
        ft = load_ft(m["imname"])
        sp = load_sp(m["imname"])
        if ft is None or sp is None:
            continue

        pred_map = run_gc_on_image(ft, sp, clf, scaler)

        valid = label_map > 0
        acc = (pred_map[valid] == label_map[valid]).mean()

        gt_rgb = np.zeros((*label_map.shape, 3), dtype=np.uint8)
        pred_rgb = np.zeros((*label_map.shape, 3), dtype=np.uint8)

        for l, c in LABEL_COLORS.items():
            gt_rgb[label_map == l] = c
            pred_rgb[pred_map == l] = c

        axes[0, col].imshow(image)
        axes[0, col].set_title(Path(m["imname"]).stem, fontsize=7)

        axes[1, col].imshow(gt_rgb)
        axes[1, col].set_title("GT", fontsize=7)

        axes[2, col].imshow(pred_rgb)
        axes[2, col].set_title(f"GC acc={acc:.2f}", fontsize=7)

        for r in range(3):
            axes[r, col].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def plot_energy_curves_gc(ds, indices, clf, scaler, save_path, n=5):
    fig, ax = plt.subplots(figsize=(6, 4))

    for idx in indices[:n]:
        _, _, m = ds[idx]
        ft = load_ft(m["imname"])
        sp = load_sp(m["imname"])
        if ft is None or sp is None:
            continue

        _, energy_curve = run_gc_on_image(
            ft, sp, clf, scaler, track_energy=True
        )

        ax.plot(
            energy_curve, marker="o", markersize=2,
            label=Path(m["imname"]).stem
        )

    ax.set_xlabel("accepted alpha-expansion move")
    ax.set_ylabel("energy")
    ax.set_title("Graph Cut / alpha-expansion energy")
    ax.legend(fontsize=6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# SAVE RESULTS
def save_results(results, out_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = out_dir / f"grid_search_gc_{timestamp}.json"
    csv_path  = out_dir / f"grid_search_gc_{timestamp}.csv"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["lambda", "beta", "accuracy"])
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\nResults saved to:\n{json_path}\n{csv_path}")


# GRID SEARCH
def hyperparameter_grid_search_gc(
    ds, test_indices, clf, scaler,
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

                preds = alpha_expansion(
                    unary, adjacency, weights,
                    lam=lam,
                    max_iters=MAX_EXPANSION_ITERS,
                    track_energy=False
                )

                pred_map = np.zeros_like(segments)
                sp_ids = np.unique(segments)
                classes = clf.classes_

                for i, sp_id in enumerate(np.sort(sp_ids)):
                    pred_map[segments == sp_id] = classes[preds[i]]

                valid = label_map > 0
                correct += (pred_map[valid] == label_map[valid]).sum()
                total += valid.sum()

            acc = correct / total if total > 0 else 0.0
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

    plt.figure(figsize=(6, 5))
    plt.imshow(grid, origin="lower")

    plt.xticks(range(len(beta_values)), [f"{b:.2f}" for b in beta_values])
    plt.yticks(range(len(lambda_values)), [f"{l:.2f}" for l in lambda_values])

    plt.xlabel("beta")
    plt.ylabel("lambda")
    plt.title("Graph Cut accuracy heatmap")

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


# MAIN
def main():
    ds = HoiemDataset(root_dir=DATASET_DIR)
    train_ds, test_ds = ds.get_split()

    train_indices = train_ds._i
    test_indices = test_ds._i

    print("Loading features...")
    with open(MODEL_DIR / "adaboost.pkl", "rb") as f:
        data = pickle.load(f)

    clf = data["clf"]
    scaler = data["scaler"]

    print("\nRunning Graph Cut inference (alpha-expansion)...")
    acc = pixel_accuracy_gc(ds, test_indices, clf, scaler)
    print(f"Pixel accuracy (Graph Cut): {acc:.4f}")

    print("\nGenerating visualizations...")

    plot_confusion_gc(
        ds, test_indices, clf, scaler,
        OUT_DIR / "graphcut_confusion.png"
    )

    plot_predictions_gc(
        ds, test_indices, clf, scaler,
        OUT_DIR / "graphcut_predictions.png"
    )

    plot_energy_curves_gc(
        ds, test_indices, clf, scaler,
        OUT_DIR / "graphcut_energy.png"
    )

    with open(MODEL_DIR / "graphcut_model.pkl", "wb") as f:
        pickle.dump({"clf": clf, "scaler": scaler}, f)


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

    lambda_values = [0.05, 0.1, 0.2, 0.4]
    beta_values   = [0.01, 0.02, 0.05, 0.1]

    print("\n=== START GRID SEARCH (GRAPH CUT) ===")

    results = hyperparameter_grid_search_gc(
        ds, test_indices, clf, scaler,
        lambda_values, beta_values,
        max_images=80
    )

    save_results(results, OUT_DIR)

    plot_results_heatmap(
        results,
        lambda_values,
        beta_values,
        OUT_DIR / "graphcut_grid_search_heatmap.png"
    )

    best = max(results, key=lambda x: x["accuracy"])

    print("\n=== BEST PARAMETERS ===")
    print(f"lambda = {best['lambda']}")
    print(f"beta   = {best['beta']}")
    print(f"acc    = {best['accuracy']:.4f}")


if __name__ == "__main__":
    main()
    # grid_search()