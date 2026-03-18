import numpy as np
import scipy.io
import cv2
from pathlib import Path


LABEL_NAMES  = {0: "unknown", 1: "ground", 2: "vertical", 3: "sky"}
LABEL_COLORS = {0: (128,128,128), 1: (46,204,113), 2: (231,76,60), 3: (52,152,219)}
LABEL_IDS    = [1, 2, 3]


class HoiemDataset:
    def __init__(self, root_dir, images_subdir="", gt_file="allimsegs2.mat",
                 split_file="rand_indices.mat", image_size=None):
        self.root       = Path(root_dir)
        self.images_dir = self.root / images_subdir if images_subdir else self.root
        self.image_size = image_size

        print(f"loading {gt_file}...")
        self._imsegs = self._load_imsegs(self.root / gt_file)
        print(f"{len(self._imsegs)} gt entries loaded")

        split_path = self.root / split_file
        self._split = self._load_split(split_path) if split_path.exists() else None

        self._img_paths    = self._find_images()
        self._valid_idx    = self._build_valid_index()
        print(f"{len(self._valid_idx)} valid pairs found\n")

    def _load_imsegs(self, path):
        mat = scipy.io.loadmat(str(path), squeeze_me=False)
        raw = mat["imsegs"].flatten()
        return [self._parse_entry(raw[i]) for i in range(len(raw))]

    @staticmethod
    def _parse_entry(entry):
        def unwrap(val):
            arr = np.array(val)
            while arr.dtype == object and arr.size == 1:
                arr = np.array(arr.flat[0])
            return arr

        d = {}
        raw = unwrap(entry["imname"])
        d["imname"]   = str(raw.flat[0]).strip()
        d["segimage"] = unwrap(entry["segimage"]).astype(np.int32)
        d["nseg"]     = int(unwrap(entry["nseg"]).flat[0])
        d["imsize"]   = unwrap(entry["imsize"]).flatten().astype(int)

        # vert_labels = main geometric label per superpixel (1=ground,2=vert,3=sky)
        d["labels"] = np.zeros(d["nseg"], dtype=np.int32)
        for candidate in ["vert_labels", "labels"]:
            try:
                arr = unwrap(entry[candidate]).flatten().astype(np.int32)
                if arr.size > 0:
                    d["labels"] = arr
                    break
            except Exception:
                continue

        for field in ["vlabels", "hlabels", "horz_labels", "adjmat"]:
            try:
                d[field] = unwrap(entry[field])
            except Exception:
                d[field] = None

        return d

    def _load_split(self, path):
        mat  = scipy.io.loadmat(str(path), squeeze_me=True)
        keys = [k for k in mat if not k.startswith("__")]
        out  = {}
        for k in keys:
            out[k] = np.array(mat[k]).flatten().astype(int) - 1  # matlab 1-indexed
        print(f"split keys: {keys}")
        return out

    def _find_images(self):
        img_map = {}
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            for p in sorted(self.images_dir.glob(ext)):
                img_map[p.stem.lower()] = p
        print(f"{len(img_map)} images found in {self.images_dir}")
        return img_map

    def _build_valid_index(self):
        valid = []
        for i, entry in enumerate(self._imsegs):
            stem = Path(entry["imname"]).stem.lower()
            if stem in self._img_paths:
                valid.append(i)
        return valid

    def __len__(self):
        return len(self._valid_idx)

    def __getitem__(self, idx):
        entry = self._imsegs[self._valid_idx[idx]]
        stem  = Path(entry["imname"]).stem.lower()
        image = self._load_image(self._img_paths[stem])

        label_map = self._build_label_map(entry)
        if image.shape[:2] != label_map.shape:
            label_map = cv2.resize(label_map, (image.shape[1], image.shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(np.int32)

        meta = {
            "imname"     : entry["imname"],
            "segimage"   : entry["segimage"],
            "nseg"       : entry["nseg"],
            "sp_labels"  : entry["labels"],
            "horz_labels": entry.get("horz_labels"),
            "adjmat"     : entry.get("adjmat"),
        }
        return image, label_map, meta

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def _load_image(self, path):
        img = cv2.imread(str(path))
        if img is None:
            raise IOError(f"cannot read {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.image_size:
            h, w = self.image_size
            img  = cv2.resize(img, (w, h))
        return img

    @staticmethod
    def _build_label_map(entry):
        seg = entry["segimage"]
        lbl = entry["labels"]
        out = np.zeros(seg.shape, dtype=np.int32)
        for sp_id in np.unique(seg):
            if 0 < sp_id <= len(lbl):
                out[seg == sp_id] = lbl[sp_id - 1]
        return out

    def get_split(self):
        inv = {v: k for k, v in enumerate(self._valid_idx)}
        if self._split is not None:
            cluster = self._split.get("cluster_images", np.array([]))
            cv      = self._split.get("cv_images", np.array([]))
            train_idx = [inv[i] for i in cluster.flatten() if i in inv]
            test_idx  = [inv[i] for i in cv.flatten()      if i in inv]
        else:
            perm      = np.random.default_rng(42).permutation(len(self)).tolist()
            train_idx, test_idx = perm[:50], perm[50:]
        print(f"split -> train: {len(train_idx)}, test: {len(test_idx)}")
        return _Subset(self, train_idx), _Subset(self, test_idx)

    def colored(self, label_map):
        rgb = np.zeros((*label_map.shape, 3), dtype=np.uint8)
        for lid, c in LABEL_COLORS.items():
            rgb[label_map == lid] = c
        return rgb

    def print_entry(self, idx=0, raw=False):
        entry = self._imsegs[idx if raw else self._valid_idx[idx]]
        print(f"\n--- entry {'raw' if raw else ''} #{idx}: {entry['imname']} ---")
        for k, v in entry.items():
            if isinstance(v, np.ndarray):
                print(f"  {k:15s}: shape={v.shape} dtype={v.dtype} unique={np.unique(v)[:6]}")
            else:
                print(f"  {k:15s}: {v}")

    def visualize(self, idx, save_path=None):
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        image, label_map, meta = self[idx]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"{meta['imname']} | nseg={meta['nseg']}", fontsize=12)
        axes[0].imshow(image);               axes[0].set_title("image");        axes[0].axis("off")
        axes[1].imshow(self.colored(label_map)); axes[1].set_title("gt");       axes[1].axis("off")
        axes[2].imshow(image)
        axes[2].imshow(self.colored(label_map), alpha=0.45)
        axes[2].set_title("overlay");        axes[2].axis("off")

        patches = [mpatches.Patch(color=[c/255 for c in LABEL_COLORS[l]], label=LABEL_NAMES[l])
                   for l in LABEL_IDS]
        fig.legend(handles=patches, loc="lower center", ncol=3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        plt.close()


class _Subset:
    def __init__(self, parent, indices):
        self._p = parent
        self._i = indices
    def __len__(self):    return len(self._i)
    def __getitem__(self, idx): return self._p[self._i[idx]]
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
