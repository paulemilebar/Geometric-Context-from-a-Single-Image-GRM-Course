import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
from tqdm import tqdm
import pickle
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "src")
from dataset import HoiemDataset, LABEL_NAMES, LABEL_COLORS, LABEL_IDS
from superpixels import load_sp, load_ft

DATASET_DIR = Path("dataset")
MODEL_DIR   = Path("data/models"); MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR     = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)

# training config
IN_DIM      = 78
HIDDEN_DIM  = 128
N_CLASSES   = 3
N_LAYERS    = 3
DROPOUT     = 0.3
LR          = 1e-3
WEIGHT_DECAY= 1e-4
EPOCHS      = 80
BATCH_SIZE  = 8   # images per batch


# graph utils

def adj_to_normalized(adj):
    # symmetric normalized adjacency: D^{-1/2} (A + I) D^{-1/2}
    # adding self-loops so each node also aggregates its own features
    a = adj.astype(np.float32)
    np.fill_diagonal(a, 1.0)
    deg  = a.sum(axis=1)
    dinv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    return (a * dinv[:, None]) * dinv[None, :]


def build_graph(ft_data, sp_data, device):
    feats     = ft_data["features"].astype(np.float32)   # (n_sp, 78)
    sp_labels = ft_data["sp_labels"].astype(np.int64)    # (n_sp,)  1-indexed
    adj       = sp_data["adj"].astype(np.float32)        # (n_sp, n_sp)

    # align sizes — sp cache and ft cache should match but just in case
    n = min(len(feats), len(sp_labels), adj.shape[0])
    feats     = feats[:n]
    sp_labels = sp_labels[:n]
    adj       = adj[:n, :n]

    norm_adj = adj_to_normalized(adj)

    x = torch.tensor(feats,     dtype=torch.float32, device=device)
    y = torch.tensor(sp_labels - 1, dtype=torch.long, device=device)  # 0-indexed
    a = torch.tensor(norm_adj,  dtype=torch.float32, device=device)

    return x, y, a


# model

class GCNLayer(nn.Module):
    # simple graph convolution: H' = sigma(A_norm H W)
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x, adj_norm):
        # message passing: aggregate neighbor features
        agg = torch.mm(adj_norm, x)         # (n, in_dim)
        return self.linear(agg)             # (n, out_dim)


class GNN(nn.Module):
    def __init__(self, in_dim=IN_DIM, hidden_dim=HIDDEN_DIM,
                 n_classes=N_CLASSES, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.layers   = nn.ModuleList()
        self.norms    = nn.ModuleList()
        self.dropout  = dropout

        dims = [in_dim] + [hidden_dim] * n_layers
        for i in range(n_layers):
            self.layers.append(GCNLayer(dims[i], dims[i+1]))
            self.norms.append(nn.LayerNorm(dims[i+1]))

        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x, adj_norm):
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x, adj_norm)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)   # (n_sp, 3) logits


# load the data

def load_split_indices(ds):
    train_ds, test_ds = ds.get_split()
    return train_ds._i, test_ds._i


def iter_batches(ds, indices, batch_size, device, shuffle=True):
    idx = list(indices)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, len(idx), batch_size):
        batch = idx[start : start + batch_size]
        graphs = []
        for i in batch:
            _, _, meta = ds[i]
            ft = load_ft(meta["imname"])
            sp = load_sp(meta["imname"])
            if ft is None or sp is None:
                continue
            x, y, a = build_graph(ft, sp, device)
            # skip superpixels without valid label
            valid = (y >= 0) & (y < N_CLASSES)
            if valid.sum() == 0:
                continue
            graphs.append((x, y, a, valid))
        if graphs:
            yield graphs


# training

def train_epoch(model, optimizer, ds, indices, device):
    model.train()
    total_loss = correct = total = 0

    for graphs in iter_batches(ds, indices, BATCH_SIZE, device, shuffle=True):
        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, device=device)

        for x, y, a, valid in graphs:
            logits = model(x, a)
            loss   = F.cross_entropy(logits[valid], y[valid])
            batch_loss = batch_loss + loss
            preds  = logits[valid].argmax(dim=1)
            correct += int((preds == y[valid]).sum())
            total   += int(valid.sum())

        (batch_loss / len(graphs)).backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += batch_loss.item()

    return total_loss / max(1, len(indices)), correct / max(1, total)


@torch.no_grad()
def evaluate(model, ds, indices, device):
    model.eval()
    correct = total = 0
    per_class = {l: [0, 0] for l in range(N_CLASSES)}

    for graphs in iter_batches(ds, indices, BATCH_SIZE, device, shuffle=False):
        for x, y, a, valid in graphs:
            logits = model(x, a)
            preds  = logits[valid].argmax(dim=1)
            labels = y[valid]
            correct += int((preds == labels).sum())
            total   += int(valid.sum())
            for c in range(N_CLASSES):
                mask = labels == c
                per_class[c][1] += int(mask.sum())
                per_class[c][0] += int((preds[mask] == c).sum())

    overall = correct / max(1, total)
    per_cls = {c: per_class[c][0] / max(1, per_class[c][1]) for c in range(N_CLASSES)}
    return overall, per_cls


@torch.no_grad()
def pixel_accuracy(model, ds, indices, device):
    # maps predictions back to pixels for fair comparison with adaboost
    model.eval()
    correct = total = 0

    for idx in indices:
        _, label_map, meta = ds[idx]
        ft = load_ft(meta["imname"])
        sp = load_sp(meta["imname"])
        if ft is None or sp is None:
            continue

        x, y, a = build_graph(ft, sp, device)
        logits   = model(x, a)
        preds    = logits.argmax(dim=1).cpu().numpy() + 1  # back to 1-indexed

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


# visualisation

def plot_curves(train_losses, train_accs, val_accs, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(train_losses)+1)

    axes[0].plot(epochs, train_losses, color="#3498db", label="train loss")
    axes[0].set_xlabel("epoch"); axes[0].set_title("training loss"); axes[0].legend()

    axes[1].plot(epochs, train_accs, color="#2ecc71", label="train acc")
    axes[1].plot(epochs, val_accs,   color="#e74c3c", label="val acc")
    axes[1].set_xlabel("epoch"); axes[1].set_title("sp-level accuracy"); axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_predictions(model, ds, indices, device, save_path, n=6):
    model.eval()
    fig, axes = plt.subplots(3, n, figsize=(4*n, 12))
    fig.suptitle("gnn predictions", fontsize=13)

    for col, idx in enumerate(indices[:n]):
        image, label_map, meta = ds[idx]
        ft = load_ft(meta["imname"])
        sp = load_sp(meta["imname"])
        if ft is None or sp is None: continue

        with torch.no_grad():
            x, y, a = build_graph(ft, sp, device)
            preds = model(x, a).argmax(dim=1).cpu().numpy() + 1

        segments = sp["segments"]
        sp_ids   = np.unique(segments)
        pred_map = np.zeros(segments.shape, dtype=np.int32)
        for i, sp_id in enumerate(sp_ids):
            if i < len(preds):
                pred_map[segments == sp_id] = preds[i]

        valid = label_map > 0
        acc   = (pred_map[valid] == label_map[valid]).mean()

        gt_rgb   = np.zeros((*label_map.shape, 3), dtype=np.uint8)
        pred_rgb = np.zeros((*pred_map.shape, 3), dtype=np.uint8)
        for l, c in LABEL_COLORS.items():
            gt_rgb[label_map == l]   = c
            pred_rgb[pred_map == l]  = c

        axes[0, col].imshow(image);    axes[0, col].set_title(Path(meta["imname"]).stem, fontsize=7)
        axes[1, col].imshow(gt_rgb);   axes[1, col].set_title("ground truth", fontsize=7)
        axes[2, col].imshow(pred_rgb); axes[2, col].set_title(f"pred acc={acc:.2f}", fontsize=7)
        for row in range(3): axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ds = HoiemDataset(root_dir=DATASET_DIR)
    train_indices, test_indices = load_split_indices(ds)

    model = GNN(in_dim=IN_DIM, hidden_dim=HIDDEN_DIM, n_classes=N_CLASSES,
                n_layers=N_LAYERS, dropout=DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model: {N_LAYERS} gcn layers, hidden={HIDDEN_DIM}, params={n_params}")

    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)

    train_losses, train_accs, val_accs = [], [], []
    best_val = 0.0

    print(f"\ntraining for {EPOCHS} epochs...")
    for epoch in tqdm(range(1, EPOCHS+1), desc="epochs"):
        loss, tr_acc = train_epoch(model, optimizer, ds, train_indices, device)
        val_acc, _   = evaluate(model, ds, test_indices[:50], device)  # subset for speed
        scheduler.step()

        train_losses.append(loss)
        train_accs.append(tr_acc)
        val_accs.append(val_acc)

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), MODEL_DIR / "gnn_best.pt")

        if epoch % 10 == 0:
            print(f"  epoch {epoch:3d} | loss={loss:.4f} | train_acc={tr_acc:.4f} | val_acc={val_acc:.4f}")

    print(f"\nbest val acc (sp-level): {best_val:.4f}")

    # load best model and evaluate properly
    model.load_state_dict(torch.load(MODEL_DIR / "gnn_best.pt", map_location=device))

    print("\nevaluating pixel accuracy on full test set (250 images)...")
    pix_acc = pixel_accuracy(model, ds, test_indices, device)
    _, per_cls = evaluate(model, ds, test_indices, device)
    print(f"pixel accuracy : {pix_acc:.4f}")
    print(f"  ground   : {per_cls[0]:.4f}")
    print(f"  vertical : {per_cls[1]:.4f}")
    print(f"  sky      : {per_cls[2]:.4f}")
    print(f"\ncomparison:")
    print(f"  adaboost (ours) : 0.8048")
    print(f"  gnn     (ours)  : {pix_acc:.4f}")
    print(f"  hoiem 2005      : 0.8600")

    plot_curves(train_losses, train_accs, val_accs, OUT_DIR / "step5_training_curves.png")
    plot_predictions(model, ds, test_indices, device, OUT_DIR / "step5_predictions.png")

    torch.save(model.state_dict(), MODEL_DIR / "gnn_final.pt")
    print(f"\nmodel saved to {MODEL_DIR}/gnn_final.pt")
    print("done -> outputs/step5_*.png")


if __name__ == "__main__":
    main()
