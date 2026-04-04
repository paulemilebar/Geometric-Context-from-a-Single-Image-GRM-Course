import sys, pickle
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import confusion_matrix
from tqdm import tqdm

sys.path.insert(0, "src")
from dataset import HoiemDataset, LABEL_NAMES, LABEL_COLORS, LABEL_IDS
from superpixels import load_sp, load_ft

DATASET_DIR  = Path("dataset")
HYPO_CACHE   = Path("data/hypotheses")
MODEL_DIR    = Path("data/models"); MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR      = Path("outputs");     OUT_DIR.mkdir(exist_ok=True)

N_EST_LABEL  = 200 
N_EST_HOMOG  = 200 
MAX_DEPTH    = 3      
MIN_PIX      = 50     



def load_hypothesis(imname):
    p = HYPO_CACHE / f"{Path(imname).stem}.npy"
    return np.load(p, allow_pickle=True).item() if p.exists() else None


def load_region_training_data(ds, train_indices):
    X, y_label, y_homog, weights = [], [], [], []
    missing = 0

    for idx in train_indices:
        _, _, meta = ds[idx]
        hd = load_hypothesis(meta["imname"])
        if hd is None:
            missing += 1
            continue

        for h in hd["hypotheses"]:
            feats     = h["region_features"]      # (nr, 78)
            labels    = h["region_labels"]         # (nr,) majority GT label
            homog     = h["region_homogeneous"]    # (nr,) bool
            pix_sizes = h["region_pixel_sizes"]   # (nr,) pixel count

            for k in range(h["n_regions"]):
                if pix_sizes[k] < MIN_PIX:
                    continue
                if labels[k] == 0:
                    continue   # région sans GT label
                X.append(feats[k])
                y_label.append(labels[k])
                y_homog.append(int(homog[k]))
                weights.append(float(pix_sizes[k]))

    if missing:
        print(f"  [warn] {missing} images sans fichier hypothèses (relancer 03b ?)")

    X       = np.array(X,       dtype=np.float32)
    y_label = np.array(y_label, dtype=np.int32)
    y_homog = np.array(y_homog, dtype=np.int32)
    weights = np.array(weights, dtype=np.float64)
    weights /= weights.mean()    

    return X, y_label, y_homog, weights


from sklearn.base import clone
import numpy as np

class ManualOneVsRest:
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
        # Utilise les probabilités pour trouver la classe majoritaire
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]


def train_label_classifier(X, y, weights):
    base_ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH),
        n_estimators=N_EST_LABEL,
        random_state=42,
    )
    
    clf = ManualOneVsRest(base_ada)

    y0 = y - 1
    clf.fit(X, y0, sample_weight=weights)
    return clf



def train_homog_classifier(X, y_homog, weights):
    clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=MAX_DEPTH),
        n_estimators=N_EST_HOMOG,
        random_state=42,
    )
    clf.fit(X, y_homog, sample_weight=weights)
    return clf


def predict_image_multiH(hd, label_clf, homog_clf, return_confidence=False):
    n_sp       = hd["n_sp"]
    confidence = np.zeros((n_sp, 3), dtype=np.float64)
    homog_sum  = np.zeros(n_sp, dtype=np.float64)

    for h in hd["hypotheses"]:
        nr_actual  = h["n_regions"]
        assignment = h["assignment"]               # (n_sp,)  feat-indexed
        reg_feats  = h["region_features"]          # (nr, 78)

        if nr_actual == 0:
            continue

        label_proba = label_clf.predict_proba(reg_feats)   # (nr, 3)

        if len(homog_clf.classes_) == 2:
            homog_proba = homog_clf.predict_proba(reg_feats)[:, 1]
        else:
            homog_proba = np.ones(nr_actual, dtype=np.float32)

        for k in range(nr_actual):
            sp_in_region          = np.where(assignment == k)[0]
            if not len(sp_in_region):
                continue
            confidence[sp_in_region] += label_proba[k]  * homog_proba[k]
            homog_sum[sp_in_region]  += homog_proba[k]

    denom = np.maximum(homog_sum, 1e-8)[:, None]
    confidence /= denom

    preds = confidence.argmax(axis=1).astype(np.int32) + 1
    if return_confidence:
        return preds, confidence
    return preds


def pixel_accuracy_multiH(ds, indices, label_clf, homog_clf):
    correct = total = 0
    per_class = {l: [0, 0] for l in LABEL_IDS}

    for idx in indices:
        _, label_map, meta = ds[idx]
        hd  = load_hypothesis(meta["imname"])
        sp  = load_sp(meta["imname"])
        ft  = load_ft(meta["imname"])
        if hd is None or sp is None or ft is None:
            continue

        sp_ids_arr = ft.get("sp_ids", np.unique(sp["segments"]))
        sp_ids_arr = np.asarray(sp_ids_arr)
        segments   = sp["segments"]
        preds      = predict_image_multiH(hd, label_clf, homog_clf)  # (n_sp,)

        pred_map = np.zeros(segments.shape, dtype=np.int32)
        for i, sp_id in enumerate(sp_ids_arr):
            if i < len(preds):
                pred_map[segments == sp_id] = preds[i]

        valid = label_map > 0
        correct += int((pred_map[valid] == label_map[valid]).sum())
        total   += int(valid.sum())

        for l in LABEL_IDS:
            lmask = (label_map == l) & valid
            per_class[l][0] += int((pred_map[lmask] == l).sum())
            per_class[l][1] += int(lmask.sum())

    overall  = correct / max(total, 1)
    per_cls  = {l: per_class[l][0] / max(per_class[l][1], 1) for l in LABEL_IDS}
    return overall, per_cls


def pixel_accuracy_oneH(ds, indices, label_clf, homog_clf, nr_target=9):
    correct = total = 0

    for idx in indices:
        _, label_map, meta = ds[idx]
        hd  = load_hypothesis(meta["imname"])
        sp  = load_sp(meta["imname"])
        ft  = load_ft(meta["imname"])
        if hd is None or sp is None or ft is None:
            continue

        #find the hypothesis with the nr closest to nr_target
        best_h = min(hd["hypotheses"], key=lambda h: abs(h["nr"] - nr_target))

        sp_ids_arr = ft.get("sp_ids", np.unique(sp["segments"]))
        sp_ids_arr = np.asarray(sp_ids_arr)
        segments   = sp["segments"]
        n_sp       = hd["n_sp"]

        reg_feats  = best_h["region_features"]
        assignment = best_h["assignment"]
        nr_actual  = best_h["n_regions"]

        label_proba = label_clf.predict_proba(reg_feats)   # (nr, 3)
        preds       = label_proba.argmax(axis=1) + 1       # (nr,) 1-indexed

        pred_map = np.zeros(segments.shape, dtype=np.int32)
        for k in range(nr_actual):
            sp_in_k = np.where(assignment == k)[0]
            for i in sp_in_k:
                if i < len(sp_ids_arr):
                    pred_map[segments == sp_ids_arr[i]] = preds[k]

        valid = label_map > 0
        correct += int((pred_map[valid] == label_map[valid]).sum())
        total   += int(valid.sum())

    return correct / max(total, 1)


def plot_predictions(ds, indices, label_clf, homog_clf, save_path, n=6):
    fig, axes = plt.subplots(3, n, figsize=(4*n, 12))
    fig.suptitle("MultiH predictions (Hoiem 2005 full pipeline)", fontsize=13)

    for col, idx in enumerate(indices[:n]):
        image, label_map, meta = ds[idx]
        hd  = load_hypothesis(meta["imname"])
        sp  = load_sp(meta["imname"])
        ft  = load_ft(meta["imname"])
        if hd is None or sp is None or ft is None:
            continue

        sp_ids_arr = ft.get("sp_ids", np.unique(sp["segments"]))
        sp_ids_arr = np.asarray(sp_ids_arr)
        segments   = sp["segments"]
        preds      = predict_image_multiH(hd, label_clf, homog_clf)

        pred_map = np.zeros(segments.shape, dtype=np.int32)
        for i, sp_id in enumerate(sp_ids_arr):
            if i < len(preds):
                pred_map[segments == sp_id] = preds[i]

        valid = label_map > 0
        acc   = (pred_map[valid] == label_map[valid]).mean() if valid.any() else 0.0

        gt_rgb   = np.zeros((*label_map.shape, 3), dtype=np.uint8)
        pred_rgb = np.zeros((*pred_map.shape, 3),  dtype=np.uint8)
        for l, c in LABEL_COLORS.items():
            gt_rgb[label_map == l]  = c
            pred_rgb[pred_map == l] = c

        axes[0, col].imshow(image)
        axes[0, col].set_title(Path(meta["imname"]).stem, fontsize=7)
        axes[1, col].imshow(gt_rgb);   axes[1, col].set_title("ground truth", fontsize=7)
        axes[2, col].imshow(pred_rgb); axes[2, col].set_title(f"MultiH acc={acc:.2f}", fontsize=7)
        for row in range(3):
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_confusion(ds, indices, label_clf, homog_clf, save_path):
    all_true, all_pred = [], []
    for idx in indices[:80]:
        _, label_map, meta = ds[idx]
        hd  = load_hypothesis(meta["imname"])
        sp  = load_sp(meta["imname"])
        ft  = load_ft(meta["imname"])
        if hd is None or sp is None or ft is None:
            continue
        sp_ids_arr = ft.get("sp_ids", np.unique(sp["segments"]))
        sp_ids_arr = np.asarray(sp_ids_arr)
        segments   = sp["segments"]
        preds      = predict_image_multiH(hd, label_clf, homog_clf)
        pred_map   = np.zeros(segments.shape, dtype=np.int32)
        for i, sp_id in enumerate(sp_ids_arr):
            if i < len(preds):
                pred_map[segments == sp_id] = preds[i]
        valid = label_map > 0
        all_true.extend(label_map[valid].tolist())
        all_pred.extend(pred_map[valid].tolist())

    cm = confusion_matrix(all_true, all_pred, labels=LABEL_IDS, normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels([LABEL_NAMES[l] for l in LABEL_IDS])
    ax.set_yticklabels([LABEL_NAMES[l] for l in LABEL_IDS])
    ax.set_xlabel("prédit"); ax.set_ylabel("vrai")
    ax.set_title("Matrice de confusion MultiH (normalisée)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                    color="white" if cm[i,j] > 0.5 else "black", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def cross_validate_multiH(ds, cv_indices, n_folds=5):
    fold_size = len(cv_indices) // n_folds
    accs = []

    for fold in range(n_folds):
        val_idx   = cv_indices[fold*fold_size : (fold+1)*fold_size]
        train_idx = cv_indices[:fold*fold_size] + cv_indices[(fold+1)*fold_size:]
        print(f"  fold {fold+1}/{n_folds} | train={len(train_idx)} val={len(val_idx)}")

        X_tr, y_lbl, y_hmg, w = load_region_training_data(ds, train_idx)
        print(f"    {len(X_tr)} régions de training "
              f"(homog={y_hmg.mean():.2f}, "
              f"grd={( y_lbl==1).mean():.2f}, "
              f"vert={(y_lbl==2).mean():.2f}, "
              f"sky={(y_lbl==3).mean():.2f})")

        lbl_clf  = train_label_classifier(X_tr, y_lbl, w)
        hmg_clf  = train_homog_classifier(X_tr, y_hmg, w)

        acc, _ = pixel_accuracy_multiH(ds, val_idx, lbl_clf, hmg_clf)
        accs.append(acc)
        print(f"    → pixel acc = {acc:.4f}")

    return accs


def main():
    ds = HoiemDataset(root_dir=DATASET_DIR)
    train_ds, test_ds = ds.get_split()
    train_indices = train_ds._i
    test_indices  = test_ds._i
    print(f"Dataset: {len(ds)} images  |  train: {len(train_indices)}  |  test: {len(test_indices)}")

    print("\nLoading region data (cluster_images)...")
    X_tr, y_lbl, y_hmg, weights = load_region_training_data(ds, train_indices)
    print(f"  {len(X_tr)} training regions loaded")
    print(f"  Label distribution : "
          f"ground={(y_lbl==1).mean():.2f}  "
          f"vert={(y_lbl==2).mean():.2f}  "
          f"sky={(y_lbl==3).mean():.2f}")
    print(f"  Homogènes : {y_hmg.mean():.2f}  |  Mixtes : {1-y_hmg.mean():.2f}")

    if len(X_tr) == 0:
        print("ERREUR : aucune région trouvée. Avez-vous lancé 03b_segmentation_hypotheses.py ?")
        return

    print(f"\nTraining label classifier  (n_est={N_EST_LABEL}, OvR AdaBoost)...")
    label_clf = train_label_classifier(X_tr, y_lbl, weights)
    print(f"  train label acc = {(label_clf.predict(X_tr)+1 == y_lbl).mean():.4f}")

    print(f"\nTraining homogeneity classifier  (n_est={N_EST_HOMOG})...")
    homog_clf = train_homog_classifier(X_tr, y_hmg, weights)
    print(f"  train homog acc = {(homog_clf.predict(X_tr) == y_hmg).mean():.4f}")

    with open(MODEL_DIR / "multiH_classifiers.pkl", "wb") as f:
        pickle.dump({"label_clf": label_clf, "homog_clf": homog_clf}, f)
    print(f"  → models saved in {MODEL_DIR}/multiH_classifiers.pkl")

    print("\nMultiH evaluation on test set (250 images)...")
    acc_multi, per_cls_multi = pixel_accuracy_multiH(
        ds, test_indices, label_clf, homog_clf)
    print(f"\n  MultiH pixel accuracy : {acc_multi:.4f}")
    print(f"    ground   : {per_cls_multi[1]:.4f}")
    print(f"    vertical : {per_cls_multi[2]:.4f}")
    print(f"    sky      : {per_cls_multi[3]:.4f}")

    print("\nOneH evaluation (nr=9, no fusion) on 100 images...")
    acc_one = pixel_accuracy_oneH(ds, test_indices[:100], label_clf, homog_clf, nr_target=9)
    print(f"  OneH pixel accuracy : {acc_one:.4f}")

    print("\n" + "═"*55)
    print("  Tableau comparatif  (Tableau 4, Hoiem 2005)")
    print("═"*55)
    print(f"  {'Méthode':<25} {'Précision pixel':>15}")
    print("-"*55)
    print(f"  {'CPrior (prior classes)':<25} {'~49 %':>15}")
    print(f"  {'Loc (position seule)':<25} {'~66 %':>15}")
    print(f"  {'Pixel (pixel level)':<25} {'~80 %':>15}")
    print(f"  {'SPixel (notre 04_adaboost)':<25} {'~83 %':>15}  ← 04_adaboost.py")
    print(f"  {'OneH (une hypothèse)':<25} {acc_one*100:>14.1f}%  ← ce script (1 hypo)")
    print(f"  {'MultiH (9 hypothèses)':<25} {acc_multi*100:>14.1f}%  ← ce script (9 hypos)")
    print(f"  {'Hoiem 2005 (papier)':<25} {'~86 %':>15}")
    print("═"*55)

    print("\n5-fold cross-validation sur les 250 cv_images (protocole Hoiem)...")
    fold_accs = cross_validate_multiH(ds, test_indices, n_folds=5)
    print(f"\n  CV résultats : {[f'{a:.4f}' for a in fold_accs]}")
    print(f"  mean = {np.mean(fold_accs):.4f}   std = {np.std(fold_accs):.4f}")
    print(f"  Hoiem 2005 : 0.8600")

    print("\nGénération des visualisations...")
    plot_predictions(ds, test_indices, label_clf, homog_clf,
                     OUT_DIR / "step4b_predictions.png")
    plot_confusion(ds, test_indices, label_clf, homog_clf,
                   OUT_DIR / "step4b_confusion.png")

    try:
        from features import FEATURE_NAMES
        importances = np.zeros(78)
        for est in label_clf.estimators_:
            importances += est.feature_importances_
        importances /= len(label_clf.estimators_)

        top_k  = 15
        top_idx = np.argsort(importances)[::-1][:top_k]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(top_k), importances[top_idx], color="#9b59b6")
        ax.set_xticks(range(top_k))
        ax.set_xticklabels([FEATURE_NAMES[i] for i in top_idx],
                           rotation=45, ha="right", fontsize=8)
        ax.set_title(f"Top {top_k} feature importances (label classifier MultiH)")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "step4b_feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  [warn] feature importance plot ignoré : {e}")

    print("done : outputs/step4b_*.png  data/models/multiH_classifiers.pkl")


if __name__ == "__main__":
    main()
