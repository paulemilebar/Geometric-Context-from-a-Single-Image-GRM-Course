"""
05d_gnn_improved_multiH.py — Improved GNN with MultiH prior.

Modified version of 05b_gnn_improved.py that uses the confidence probabilities
from the 9-hypotheses fusion model (04b_multiH_classify.py) instead of the 
simple SPixel AdaBoost model.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
import importlib
import pickle
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from dataset import HoiemDataset, LABEL_NAMES, LABEL_COLORS, LABEL_IDS
from superpixels import load_sp, load_ft

DATASET_DIR = Path("dataset")
MODEL_DIR   = Path("data/models"); MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR     = Path("outputs");     OUT_DIR.mkdir(exist_ok=True)

# ── hyper-parameters ──────────────────────────────────────────────────────────
BASE_DIM     = 78          # raw superpixel features
ADA_DIM      = 3           # appended AdaBoost class probabilities
IN_DIM       = BASE_DIM + ADA_DIM   # 81
HIDDEN_DIM   = 256
N_CLASSES    = 3
N_LAYERS     = 4
N_HEADS      = 4           # must divide HIDDEN_DIM evenly
DROPOUT      = 0.25
LR           = 3e-4
WEIGHT_DECAY = 5e-4
EPOCHS       = 150
BATCH_SIZE   = 16
LABEL_SMOOTH = 0.05

from sklearn.base import clone

class ManualOneVsRest:
    """Un OneVsRest personnalisé qui ignore les bugs de routage de scikit-learn 1.4+"""
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


# ── MultiH prior feature injection ─────────────────────────────────────────

try:
    multiH = importlib.import_module("04b_multiH_classify")
    predict_image_multiH = multiH.predict_image_multiH
    load_hypothesis = multiH.load_hypothesis
except Exception as e:
    print(f"[err] Could not import 04b_multiH_classify: {e}")

def load_multih():
    """Load MultiH classifiers from disk."""
    p = MODEL_DIR / "multiH_classifiers.pkl"
    if not p.exists():
        print("[warn] multiH_classifiers.pkl not found — using zeros for class-prob features")
        return None, None
    with open(p, "rb") as f:
        d = pickle.load(f)
    return d["label_clf"], d["homog_clf"]

LBL_CLF, HOMOG_CLF = load_multih()


def augment_features(feats: np.ndarray, imname: str) -> np.ndarray:
    """
    feats : (n_sp, 78) float32
    Returns (n_sp, 81) — 78 raw features + 3 MultiH class probabilities.
    """
    if LBL_CLF is None:
        proba = np.zeros((len(feats), ADA_DIM), dtype=np.float32)
    else:
        hd = load_hypothesis(imname)
        if hd is None:
            proba = np.zeros((len(feats), ADA_DIM), dtype=np.float32)
        else:
            _, proba = predict_image_multiH(hd, LBL_CLF, HOMOG_CLF, return_confidence=True)
            proba = proba.astype(np.float32)
            proba = proba[:len(feats)]
    return np.concatenate([feats, proba], axis=1)


# ── graph utilities ───────────────────────────────────────────────────────────

def adj_to_edge_index(adj: np.ndarray, device):
    """
    adj : (N, N) binary adjacency matrix.
    Returns (src, dst) as LongTensors on device (self-loops included).
    """
    a = adj.copy().astype(np.uint8)
    np.fill_diagonal(a, 1)
    rows, cols = np.where(a > 0)
    src = torch.tensor(rows, dtype=torch.long, device=device)
    dst = torch.tensor(cols, dtype=torch.long, device=device)
    return src, dst


def build_graph(ft_data: dict, sp_data: dict, meta: dict, device):
    feats     = ft_data["features"].astype(np.float32)    # (n_sp, 78)
    sp_labels = ft_data["sp_labels"].astype(np.int64)     # (n_sp,)
    adj       = sp_data["adj"].astype(np.float32)         # (n_sp, n_sp)

    n         = min(len(feats), len(sp_labels), adj.shape[0])
    feats     = feats[:n]
    sp_labels = sp_labels[:n]
    adj       = adj[:n, :n]

    feats81   = augment_features(feats, meta["imname"])   # (n, 81)

    x         = torch.tensor(feats81,       dtype=torch.float32, device=device)
    y         = torch.tensor(sp_labels - 1, dtype=torch.long,    device=device)
    src, dst  = adj_to_edge_index(adj, device)
    return x, y, src, dst


# ── model ─────────────────────────────────────────────────────────────────────

class GATLayer(nn.Module):
    """
    Multi-head Graph Attention layer — pure PyTorch, no PyG/scatter needed.

    Implements:
        e_ij  = LeakyReLU( a_src · W h_i  +  a_dst · W h_j )
        α_ij  = softmax_j( e_ij )           (per destination node)
        h'_i  = concat_k [ Σ_j α^k_ij  W^k h_j ]
    """
    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, dropout: float = 0.25):
        super().__init__()
        assert out_dim % n_heads == 0, "out_dim must be divisible by n_heads"
        self.n_heads  = n_heads
        self.head_dim = out_dim // n_heads

        self.W     = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(n_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(n_heads, self.head_dim))
        nn.init.xavier_normal_(self.a_src.unsqueeze(0))
        nn.init.xavier_normal_(self.a_dst.unsqueeze(0))

        self.leaky   = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, src_idx: torch.Tensor, dst_idx: torch.Tensor):
        N  = x.size(0)
        Wx = self.W(x).view(N, self.n_heads, self.head_dim)   # (N, H, D)

        # Attention scores  (E, H)
        e_src = (Wx[src_idx] * self.a_src).sum(dim=-1)
        e_dst = (Wx[dst_idx] * self.a_dst).sum(dim=-1)
        e     = self.leaky(e_src + e_dst)

        # Stable softmax per destination node
        # Subtract global max per head for numerical stability
        e_shifted = e - e.max(dim=0, keepdim=True)[0]        # (E, H)
        exp_e     = torch.exp(e_shifted)                       # (E, H)
        exp_sum   = torch.zeros(N, self.n_heads, device=x.device)
        exp_sum.index_add_(0, dst_idx, exp_e)
        alpha = exp_e / (exp_sum[dst_idx] + 1e-8)             # (E, H)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # Weighted message aggregation
        msg = (Wx[src_idx] * alpha.unsqueeze(-1)).reshape(-1, self.n_heads * self.head_dim)
        out = torch.zeros(N, self.n_heads * self.head_dim, device=x.device)
        out.index_add_(0, dst_idx, msg)                        # (N, out_dim)
        return out


class GATBlock(nn.Module):
    """GAT layer + residual connection + LayerNorm + ELU activation."""
    def __init__(self, dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.gat  = GATLayer(dim, dim, n_heads, dropout)
        self.norm = nn.LayerNorm(dim)
        self.act  = nn.ELU()
        self.drop = dropout

    def forward(self, x, src_idx, dst_idx):
        h = self.gat(x, src_idx, dst_idx)
        h = F.dropout(h, p=self.drop, training=self.training)
        return self.act(self.norm(x + h))          # residual


class ImprovedGNN(nn.Module):
    def __init__(self, in_dim: int = IN_DIM, hidden_dim: int = HIDDEN_DIM,
                 n_classes: int = N_CLASSES, n_layers: int = N_LAYERS,
                 n_heads: int = N_HEADS, dropout: float = DROPOUT):
        super().__init__()
        # Project raw features into the hidden space
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )
        # Stacked GAT blocks (constant width → easy residuals)
        self.blocks = nn.ModuleList([
            GATBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x, src_idx, dst_idx):
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x, src_idx, dst_idx)
        return self.classifier(x)                  # (N, 3) logits


# ── data loading ──────────────────────────────────────────────────────────────

def load_split_indices(ds):
    train_ds, test_ds = ds.get_split()
    return train_ds._i, test_ds._i


def iter_batches(ds, indices, batch_size, device, shuffle=True):
    idx = list(indices)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        batch  = idx[start : start + batch_size]
        graphs = []
        for i in batch:
            _, _, meta = ds[i]
            ft = load_ft(meta["imname"])
            sp = load_sp(meta["imname"])
            if ft is None or sp is None:
                continue
            x, y, src, dst = build_graph(ft, sp, meta, device)
            valid = (y >= 0) & (y < N_CLASSES)
            if valid.sum() == 0:
                continue
            graphs.append((x, y, src, dst, valid))
        if graphs:
            yield graphs


# ── class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(ds, indices, device):
    """Compute inverse-frequency class weights from training superpixel labels."""
    counts = np.zeros(N_CLASSES, dtype=np.float64)
    for idx in indices:
        _, _, meta = ds[idx]
        ft = load_ft(meta["imname"])
        if ft is None:
            continue
        labels = ft["sp_labels"].astype(np.int64)
        for c in range(N_CLASSES):
            counts[c] += (labels == c + 1).sum()    # 1-indexed labels
    # inverse frequency, normalized so mean weight = 1
    inv = 1.0 / (counts + 1e-8)
    weights = inv / inv.mean()
    print(f"class weights: ground={weights[0]:.3f}  vert={weights[1]:.3f}  sky={weights[2]:.3f}")
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ── training ──────────────────────────────────────────────────────────────────

def train_epoch(model, optimizer, ds, indices, device, class_weights=None):
    model.train()
    total_loss = correct = total = 0

    for graphs in iter_batches(ds, indices, BATCH_SIZE, device, shuffle=True):
        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, device=device)

        for x, y, src, dst, valid in graphs:
            logits     = model(x, src, dst)
            loss       = F.cross_entropy(logits[valid], y[valid],
                                         weight=class_weights,
                                         label_smoothing=LABEL_SMOOTH)
            batch_loss = batch_loss + loss
            preds      = logits[valid].argmax(dim=1)
            correct   += int((preds == y[valid]).sum())
            total     += int(valid.sum())

        (batch_loss / len(graphs)).backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += batch_loss.item()

    return total_loss / max(1, len(indices)), correct / max(1, total)


@torch.no_grad()
def evaluate(model, ds, indices, device):
    model.eval()
    correct = total = 0
    per_class = {c: [0, 0] for c in range(N_CLASSES)}

    for graphs in iter_batches(ds, indices, BATCH_SIZE, device, shuffle=False):
        for x, y, src, dst, valid in graphs:
            logits = model(x, src, dst)
            preds  = logits[valid].argmax(dim=1)
            labels = y[valid]
            correct += int((preds == labels).sum())
            total   += int(valid.sum())
            for c in range(N_CLASSES):
                m = labels == c
                per_class[c][1] += int(m.sum())
                per_class[c][0] += int((preds[m] == c).sum())

    overall = correct / max(1, total)
    per_cls = {c: per_class[c][0] / max(1, per_class[c][1]) for c in range(N_CLASSES)}
    return overall, per_cls


@torch.no_grad()
def pixel_accuracy(model, ds, indices, device):
    """Pixel-level accuracy — fair comparison with AdaBoost."""
    model.eval()
    correct = total = 0

    for idx in indices:
        _, label_map, meta = ds[idx]
        ft = load_ft(meta["imname"])
        sp = load_sp(meta["imname"])
        if ft is None or sp is None:
            continue

        x, y, src, dst = build_graph(ft, sp, meta, device)
        preds    = model(x, src, dst).argmax(dim=1).cpu().numpy() + 1  # 1-indexed

        segments = sp["segments"]
        sp_ids   = np.unique(segments)
        pred_map = np.zeros(segments.shape, dtype=np.int32)
        for i, sp_id in enumerate(sp_ids):
            if i < len(preds):
                pred_map[segments == sp_id] = preds[i]

        valid    = label_map > 0
        correct += int((pred_map[valid] == label_map[valid]).sum())
        total   += int(valid.sum())

    return correct / max(1, total)


# ── visualisation ─────────────────────────────────────────────────────────────

def plot_curves(train_losses, train_accs, val_pix_accs, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, len(train_losses) + 1)

    axes[0].plot(ep, train_losses, color="#3498db", label="train loss")
    axes[0].set_xlabel("epoch"); axes[0].set_title("training loss"); axes[0].legend()

    axes[1].plot(ep, train_accs,   color="#2ecc71", label="train sp-acc")
    axes[1].plot(ep, val_pix_accs, color="#e74c3c", label="val pixel-acc (subset)")
    axes[1].axhline(0.86, color="gray", ls=":", label="hoiem 2005 (0.86)")
    axes[1].axhline(0.7906, color="orange", ls="--", label="adaboost test (0.79)")
    axes[1].set_xlabel("epoch"); axes[1].set_title("accuracy"); axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_predictions(model, ds, indices, device, save_path, n=6):
    model.eval()
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 12))
    fig.suptitle("improved GNN predictions", fontsize=13)

    for col, idx in enumerate(indices[:n]):
        image, label_map, meta = ds[idx]
        ft = load_ft(meta["imname"])
        sp = load_sp(meta["imname"])
        if ft is None or sp is None:
            continue

        with torch.no_grad():
            x, y, src, dst = build_graph(ft, sp, meta, device)
            preds = model(x, src, dst).argmax(dim=1).cpu().numpy() + 1

        segments = sp["segments"]
        sp_ids   = np.unique(segments)
        pred_map = np.zeros(segments.shape, dtype=np.int32)
        for i, sp_id in enumerate(sp_ids):
            if i < len(preds):
                pred_map[segments == sp_id] = preds[i]

        valid = label_map > 0
        acc   = (pred_map[valid] == label_map[valid]).mean()

        gt_rgb   = np.zeros((*label_map.shape, 3), dtype=np.uint8)
        pred_rgb = np.zeros((*pred_map.shape, 3),  dtype=np.uint8)
        for l, c in LABEL_COLORS.items():
            gt_rgb[label_map == l]  = c
            pred_rgb[pred_map == l] = c

        axes[0, col].imshow(image);    axes[0, col].set_title(Path(meta["imname"]).stem, fontsize=7)
        axes[1, col].imshow(gt_rgb);   axes[1, col].set_title("ground truth", fontsize=7)
        axes[2, col].imshow(pred_rgb); axes[2, col].set_title(f"pred acc={acc:.2f}", fontsize=7)
        for row in range(3):
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ds = HoiemDataset(root_dir=DATASET_DIR)
    train_indices, test_indices = load_split_indices(ds)
    val_subset = test_indices[:30]      # fast subset for mid-training eval
    print(f"train: {len(train_indices)} images (cluster_images) | test: {len(test_indices)} images (cv_images)")

    model = ImprovedGNN().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model : {N_LAYERS} GAT blocks | hidden={HIDDEN_DIM} | heads={N_HEADS} | params={n_params:,}")
    print(f"input : {IN_DIM}-dim  (78 raw features + 3 MultiH class probs)")
    print(f"multiH prior : {'loaded ✓' if LBL_CLF is not None else 'NOT found — using zero proba features'}")

    class_weights = compute_class_weights(ds, train_indices, device)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * 0.01)

    train_losses, train_accs, val_pix_accs = [], [], []
    best_val = 0.0

    print(f"\ntraining for {EPOCHS} epochs on {len(train_indices)} images...")
    for epoch in tqdm(range(1, EPOCHS + 1), desc="epochs"):
        loss, tr_acc = train_epoch(model, optimizer, ds, train_indices, device,
                                   class_weights=class_weights)
        scheduler.step()

        # Pixel-level val on 30-image subset every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            val_pix = pixel_accuracy(model, ds, val_subset, device)
        else:
            val_pix = val_pix_accs[-1] if val_pix_accs else 0.0

        train_losses.append(loss)
        train_accs.append(tr_acc)
        val_pix_accs.append(val_pix)

        if val_pix > best_val:
            best_val = val_pix
            torch.save(model.state_dict(), MODEL_DIR / "gnn_improved_multiH_best.pt")

        if epoch % 10 == 0:
            tqdm.write(
                f"  epoch {epoch:3d} | loss={loss:.4f} | "
                f"train_acc={tr_acc:.4f} | val_pix={val_pix:.4f}"
            )

    print(f"\nbest val pixel acc (30-img subset): {best_val:.4f}")

    # ── final evaluation ──────────────────────────────────────────────────────
    model.load_state_dict(
        torch.load(MODEL_DIR / "gnn_improved_multiH_best.pt", map_location=device)
    )

    print("\nevaluating pixel accuracy on full test set (250 images)...")
    pix_acc = pixel_accuracy(model, ds, test_indices, device)
    _, per_cls = evaluate(model, ds, test_indices, device)

    print(f"\npixel accuracy : {pix_acc:.4f}")
    print(f"  ground   : {per_cls[0]:.4f}")
    print(f"  vertical : {per_cls[1]:.4f}")
    print(f"  sky      : {per_cls[2]:.4f}")
    print(f"\ncomparison:")
    print(f"  adaboost (ours)      : 0.7906 (train=50)  / 0.8119 (5-fold CV)")
    print(f"  gnn basic (05_gnn)   : see outputs/step5_*.png")
    print(f"  gnn improved (ours)  : {pix_acc:.4f}")
    print(f"  hoiem 2005           : 0.8600")

    plot_curves(train_losses, train_accs, val_pix_accs,
                OUT_DIR / "step5d_training_curves.png")
    plot_predictions(model, ds, test_indices, device,
                     OUT_DIR / "step5d_predictions.png")

    torch.save(model.state_dict(), MODEL_DIR / "gnn_improved_multiH_final.pt")
    print(f"\nmodel saved to {MODEL_DIR}/gnn_improved_multiH_final.pt")
    print("done -> outputs/step5d_*.png")


if __name__ == "__main__":
    main()
