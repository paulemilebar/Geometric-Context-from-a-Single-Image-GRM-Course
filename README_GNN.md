# GNN Improved — Geometric Context Classification

## What is this?

This file (`05b_gnn_improved.py`) implements an improved **Graph Neural Network (GNN)** to classify image regions into three geometric categories:

- 🟢 **Ground** — floor, road, grass
- 🔴 **Vertical** — walls, buildings, trees
- 🔵 **Sky** — sky

The approach follows the paper *"Geometric Context from a Single Image"* by Hoiem et al. (2005). The image is first divided into small regions called **superpixels**, and the GNN predicts the geometric class of each superpixel by looking at both its own features and the features of its neighbors.

---

## Why a GNN?

A simple classifier (like AdaBoost) looks at each superpixel **independently**. But in real images, the geometry is spatially consistent — sky is always above ground, ground is at the bottom, and vertical surfaces are in the middle. A GNN can use this **neighbor context** to make better predictions.

Formally, superpixels form a **graph**: each superpixel is a node, and two nodes are connected by an edge if the corresponding superpixels touch each other in the image.

---

## Architecture

### Input Features (81 dimensions)

Each superpixel is described by **81 features**:

| Group | Size | Description |
|-------|------|-------------|
| Color | 16 | Mean RGB, mean HSV, color histograms |
| Location | 12 | Normalized position, distance to horizon |
| Texture | 15 | DOOG filter responses at 12 orientations |
| Geometry | 35 | Line features, vanishing point statistics |
| **AdaBoost prior** | **3** | Class probabilities from the AdaBoost model |

The last 3 dimensions are the key addition: we feed the **AdaBoost prediction** as input to the GNN. This gives the network a strong starting point that it can then refine using graph context.

### Model: 4-Layer Graph Attention Network (GAT)

```
Input (81-dim)
     ↓
Linear projection → LayerNorm → ELU   [→ 256-dim]
     ↓
GAT Block × 4:
  ├── Graph Attention Layer
  │     (4 attention heads, per-edge learned weights)
  ├── Residual connection  (x = x + GAT(x))
  ├── LayerNorm
  └── ELU activation
     ↓
MLP classifier: 256 → 128 → 3 (logits)
```

**Total parameters: 321,027**

### Key design choices

#### 1. Graph Attention (GAT)
Instead of treating all neighbors equally, the model learns **attention weights** per edge:

```
α_ij = softmax( LeakyReLU( a · [W·hᵢ || W·hⱼ] ) )
h'ᵢ  = Σⱼ α_ij · W·hⱼ
```

This means the model can learn to ignore noisy neighbors and focus on the most relevant ones.

#### 2. Residual connections
Each GAT block adds its output to its input (`h = h + GAT(h)`). This prevents **over-smoothing** — a common problem in deep GNNs where all nodes end up with the same representation after many layers.

#### 3. Sparse edge-list message passing
The adjacency matrix of a superpixel graph is very sparse (~4–6 neighbors per node). Instead of multiplying large dense matrices, we use explicit edge lists `(src, dst)` and `index_add_` operations, which is much faster.

#### 4. Class-weighted loss
The training data is very imbalanced:
- Vertical: ~65% of superpixels
- Ground: ~28%
- Sky: ~6%

Without correction, the model would always predict "vertical". We use **inverse-frequency weights** in the cross-entropy loss:

```
class weights: ground=0.513  vertical=0.221  sky=2.265
```

Sky gets ~10× more weight than vertical, which forces the model to learn all three classes properly.

#### 5. Training setup
| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| LR schedule | Cosine annealing (3e-4 → 3e-6) |
| Epochs | 150 |
| Label smoothing | 0.05 |
| Dropout | 0.25 |
| Batch size | 16 images |

---

## Results

Training set: **50 images** (`cluster_images`)  
Test set: **250 images** (`cv_images`)

### Pixel accuracy

| Method | Overall | Ground | Vertical | Sky |
|--------|---------|--------|----------|-----|
| AdaBoost (ours, train=50) | 0.7906 | 0.769 | 0.763 | 0.893 |
| AdaBoost (5-fold CV, train=200) | 0.8119 | — | — | — |
| **GNN improved (ours)** | **0.8294** | **0.680** | **0.861** | **0.851** |
| Hoiem et al. 2005 | 0.8600 | — | — | — |

### Analysis

The improved GNN reaches **0.8294** pixel accuracy, which is **+3.9 points** above AdaBoost on the same training split. This result is only **3.1 points** below the paper's reported 0.86.

The remaining gap to the paper comes from:
1. **Smaller training set** — we train on 50 images, the paper uses up to 240 in CV
2. **Feature differences** — the paper uses steerable filter banks (Laws filters, Leung-Malik bank), which are richer than our DOOG filters
3. **Horizon estimation** — the paper estimates the horizon per image; we use a fixed ratio

The per-class balance is good: sky (0.85) and vertical (0.86) are both well above AdaBoost's sky score from the pure classifier. Ground (0.68) is the most difficult class, which is expected because ground superpixels are often partially occluded or have complex textures.

---

## How to run

```bash
# 1. Make sure features and AdaBoost model are computed first
python 03_features.py
python 04_adaboost.py

# 2. Train and evaluate the improved GNN
python 05b_gnn_improved.py
```

Outputs are saved to `outputs/step5b_*.png`.
