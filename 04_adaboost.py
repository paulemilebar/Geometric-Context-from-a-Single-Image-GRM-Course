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
from tqdm import tqdm

sys.path.insert(0, "src")
from dataset import HoiemDataset, LABEL_NAMES, LABEL_COLORS, LABEL_IDS
from superpixels import load_sp, load_ft

DATASET_DIR = Path("dataset")
FT_CACHE    = Path("data/features")
SP_CACHE    = Path("data/superpixels")
MODEL_DIR   = Path("data/models"); MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR     = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)

# adaboost params following hoiem 2005 (logistic regression adaboost, 8-node trees)
N_ESTIMATORS = 200
MAX_DEPTH    = 3   # ~8 leaf nodes (paper: 8-node decision trees)


def load_split_data(ds, indices):
    X, y, meta = [], [], []
    for idx in indices:
        _, _, m = ds[idx]
        ft = load_ft(m["imname"])
        if ft is None:
            continue
        feats     = ft["features"]    # (n_sp, 78)
        sp_labels = ft["sp_labels"]   # (n_sp,)
        # keep only labeled superpixels
        mask = sp_labels > 0
        X.append(feats[mask])
        y.append(sp_labels[mask])
        meta.append({"imname": m["imname"], "n_sp": mask.sum()})
    return np.vstack(X), np.concatenate(y), meta


def pixel_accuracy(ds, indices, clf, scaler):
    correct = total = 0
    per_class = {l: [0, 0] for l in LABEL_IDS}  # [correct, total]

    for idx in indices:
        _, label_map, m = ds[idx]
        ft = load_ft(m["imname"])
        sp = load_sp(m["imname"])
        if ft is None or sp is None:
            continue

        feats    = ft["features"]
        segments = sp["segments"]
        sp_ids   = np.unique(segments)

        X = scaler.transform(feats)
        preds = clf.predict(X)   # (n_sp,)

        # map superpixel predictions back to pixels
        pred_map = np.zeros(segments.shape, dtype=np.int32)
        for i, sp_id in enumerate(sp_ids):
            pred_map[segments == sp_id] = preds[i]

        valid = label_map > 0
        correct += int((pred_map[valid] == label_map[valid]).sum())
        total   += int(valid.sum())

        for l in LABEL_IDS:
            lmask = (label_map == l) & valid
            per_class[l][0] += int((pred_map[lmask] == l).sum())
            per_class[l][1] += int(lmask.sum())

    overall = correct / total if total else 0.0
    per_cls = {l: (per_class[l][0] / per_class[l][1]) if per_class[l][1] else 0.0
               for l in LABEL_IDS}
    return overall, per_cls


def cross_validate(ds, cv_indices, n_folds=5):
    # cv_indices are grouped in blocks of 50 (hoiem convention)
    fold_size = len(cv_indices) // n_folds
    accs = []
    per_cls_folds = []   # collect per-class acc per fold

    for fold in range(n_folds):
        val_idx   = cv_indices[fold*fold_size : (fold+1)*fold_size]
        train_idx = cv_indices[:fold*fold_size] + cv_indices[(fold+1)*fold_size:]

        print(f"  fold {fold+1}/{n_folds} | train={len(train_idx)} val={len(val_idx)}")

        X_tr, y_tr, _ = load_split_data(ds, train_idx)
        scaler = StandardScaler().fit(X_tr)
        X_tr   = scaler.transform(X_tr)

        clf = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH),
            n_estimators=N_ESTIMATORS,
            random_state=42,
        )
        clf.fit(X_tr, y_tr)

        acc, per_cls = pixel_accuracy(ds, val_idx, clf, scaler)
        accs.append(acc)
        per_cls_folds.append(per_cls)
        print(f"    pixel acc: {acc:.4f}  "
              f"ground={per_cls[1]:.3f}  vert={per_cls[2]:.3f}  sky={per_cls[3]:.3f}")

    # average per-class accuracies across folds
    avg_per_cls = {l: np.mean([f[l] for f in per_cls_folds]) for l in LABEL_IDS}
    return accs, avg_per_cls


def plot_confusion(ds, indices, clf, scaler, save_path):
    all_preds, all_true = [], []
    for idx in indices[:50]:   # subset for speed
        _, label_map, m = ds[idx]
        ft = load_ft(m["imname"])
        sp = load_sp(m["imname"])
        if ft is None or sp is None: continue

        segments = sp["segments"]
        sp_ids   = np.unique(segments)
        X = scaler.transform(ft["features"])
        preds = clf.predict(X)

        pred_map = np.zeros(segments.shape, dtype=np.int32)
        for i, sp_id in enumerate(sp_ids):
            pred_map[segments == sp_id] = preds[i]

        valid = label_map > 0
        all_true.extend(label_map[valid].tolist())
        all_preds.extend(pred_map[valid].tolist())

    cm = confusion_matrix(all_true, all_preds, labels=LABEL_IDS, normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels([LABEL_NAMES[l] for l in LABEL_IDS])
    ax.set_yticklabels([LABEL_NAMES[l] for l in LABEL_IDS])
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("confusion matrix (normalized)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                    color="white" if cm[i,j] > 0.5 else "black", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_predictions(ds, indices, clf, scaler, save_path, n=6):
    fig, axes = plt.subplots(3, n, figsize=(4*n, 12))
    fig.suptitle("adaboost predictions", fontsize=13)

    for col, idx in enumerate(indices[:n]):
        image, label_map, m = ds[idx]
        ft = load_ft(m["imname"])
        sp = load_sp(m["imname"])
        if ft is None or sp is None: continue

        segments = sp["segments"]
        sp_ids   = np.unique(segments)
        X = scaler.transform(ft["features"])
        preds = clf.predict(X)

        pred_map = np.zeros(segments.shape, dtype=np.int32)
        for i, sp_id in enumerate(sp_ids):
            pred_map[segments == sp_id] = preds[i]

        # accuracy for this image
        valid = label_map > 0
        acc = (pred_map[valid] == label_map[valid]).mean()

        gt_rgb   = np.zeros((*label_map.shape, 3), dtype=np.uint8)
        pred_rgb = np.zeros((*label_map.shape, 3), dtype=np.uint8)
        for l, c in LABEL_COLORS.items():
            gt_rgb[label_map == l]   = c
            pred_rgb[pred_map == l]  = c

        axes[0, col].imshow(image)
        axes[0, col].set_title(Path(m["imname"]).stem, fontsize=7)
        axes[1, col].imshow(gt_rgb);   axes[1, col].set_title("ground truth", fontsize=7)
        axes[2, col].imshow(pred_rgb); axes[2, col].set_title(f"pred acc={acc:.2f}", fontsize=7)
        for row in range(3): axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    ds = HoiemDataset(root_dir=DATASET_DIR)
    train_ds, test_ds = ds.get_split()

    train_indices = train_ds._i
    test_indices  = test_ds._i

    # 1) train final model on cluster_images (50 images)
    print("loading training features...")
    X_tr, y_tr, _ = load_split_data(ds, train_indices)
    print(f"train set: {X_tr.shape[0]} superpixels, {X_tr.shape[1]} features")
    print(f"class dist: ground={( y_tr==1).mean():.2f}  vert={(y_tr==2).mean():.2f}  sky={(y_tr==3).mean():.2f}")

    scaler = StandardScaler().fit(X_tr)
    X_tr   = scaler.transform(X_tr)

    print(f"\ntraining adaboost (n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH})...")
    clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH),
        n_estimators=N_ESTIMATORS,
        random_state=42,
    )
    clf.fit(X_tr, y_tr)
    print("done")

    # 2) evaluate on test set (pixel accuracy)
    print("\nevaluating pixel accuracy on test set (250 images)...")
    acc, per_cls = pixel_accuracy(ds, test_indices, clf, scaler)
    print(f"\noverall pixel accuracy : {acc:.4f}")
    print(f"  ground   : {per_cls[1]:.4f}")
    print(f"  vertical : {per_cls[2]:.4f}")
    print(f"  sky      : {per_cls[3]:.4f}")
    print(f"\n  hoiem 2005 baseline  : 0.8600")

    # 3a) 5-fold cross-validation on cv_images only (250 images) — current setup
    print("\n5-fold cross-validation on cv_images (250 images)...")
    cv_global = test_indices
    fold_accs, cv_per_cls = cross_validate(ds, cv_global, n_folds=5)
    print(f"\ncv results: {[f'{a:.4f}' for a in fold_accs]}")
    print(f"mean={np.mean(fold_accs):.4f}  std={np.std(fold_accs):.4f}")
    print(f"  cv per-class avg:  ground={cv_per_cls[1]:.4f}  vert={cv_per_cls[2]:.4f}  sky={cv_per_cls[3]:.4f}")

    # 3b) 5-fold CV on ALL 300 images — matches hoiem 2005 protocol (240 train / 60 val)
    print("\n5-fold cross-validation on ALL 300 images (hoiem protocol)...")
    all_indices = list(range(len(ds)))
    fold_accs_all, cv300_per_cls = cross_validate(ds, all_indices, n_folds=5)
    print(f"\ncv300 results: {[f'{a:.4f}' for a in fold_accs_all]}")
    print(f"mean={np.mean(fold_accs_all):.4f}  std={np.std(fold_accs_all):.4f}")
    print(f"  cv300 per-class avg:  ground={cv300_per_cls[1]:.4f}  vert={cv300_per_cls[2]:.4f}  sky={cv300_per_cls[3]:.4f}")
    print(f"  hoiem 2005 baseline  : 0.8600")

    # 4) feature importance
    importances = clf.feature_importances_
    top_k = 15
    top_idx = np.argsort(importances)[::-1][:top_k]
    from features import FEATURE_NAMES
    print(f"\ntop {top_k} features:")
    for rank, fi in enumerate(top_idx):
        print(f"  {rank+1:2d}. {FEATURE_NAMES[fi]:15s} {importances[fi]:.4f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(top_k), importances[top_idx], color="#3498db")
    ax.set_xticks(range(top_k))
    ax.set_xticklabels([FEATURE_NAMES[i] for i in top_idx], rotation=45, ha="right", fontsize=8)
    ax.set_title(f"top {top_k} feature importances")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "step4_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 5) confusion matrix and predictions
    plot_confusion(ds, test_indices, clf, scaler, OUT_DIR / "step4_confusion.png")
    plot_predictions(ds, test_indices, clf, scaler, OUT_DIR / "step4_predictions.png")

    # 6) cv accuracy curve (accuracy vs n_estimators)
    print("\ncomputing accuracy vs n_estimators...")
    _, label_map_0, m0 = ds[test_indices[0]]
    ft0 = load_ft(m0["imname"]); sp0 = load_sp(m0["imname"])
    steps = range(10, N_ESTIMATORS+1, 10)
    staged_accs = []
    for n in steps:
        sub_clf = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH),
            n_estimators=n, random_state=42
        )
        sub_clf.fit(X_tr, y_tr)
        a, _ = pixel_accuracy(ds, test_indices[:30], sub_clf, scaler)
        staged_accs.append(a)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(list(steps), staged_accs, marker="o", markersize=3, color="#2ecc71")
    ax.axhline(acc, color="red", ls="--", label=f"final ({acc:.3f})")
    ax.axhline(0.86, color="gray", ls=":", label="hoiem 2005 (0.86)")
    ax.set_xlabel("n_estimators"); ax.set_ylabel("pixel accuracy")
    ax.set_title("accuracy vs number of estimators")
    ax.legend(); plt.tight_layout()
    plt.savefig(OUT_DIR / "step4_learning_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # save model
    with open(MODEL_DIR / "adaboost.pkl", "wb") as f:
        pickle.dump({"clf": clf, "scaler": scaler}, f)
    print(f"\nmodel saved to {MODEL_DIR}/adaboost.pkl")
    print("done -> outputs/step4_*.png")


if __name__ == "__main__":
    main()
