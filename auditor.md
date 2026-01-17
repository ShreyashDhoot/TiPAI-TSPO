# 📘 Auditor.ipynb — Risk Heatmap Auditor

**Repository:** TiPAI-TSPO  
**Notebook:** `Auditor.ipynb`  
**Author:** Shreyash Dhoot  
**Domain:** Vision • Weak Supervision • Model Auditing • Explainability

---

## 🚀 Overview

`Auditor.ipynb` implements a **weakly supervised risk-auditing pipeline for images**. The notebook trains a lightweight **risk prediction head** on top of a **pretrained CLIP-based ResNet50 backbone**, and produces **spatial risk heatmaps** that explain *where* the model focuses when estimating risk.

The core idea is:

> Instead of only predicting a label, **audit the model’s attention** by learning patch-level risk signals under weak (global) supervision.

This is useful for:
- Model interpretability
- Dataset inspection
- Weakly supervised localization
- Risk-sensitive or safety-critical applications

---

## 🎯 Key Objectives

1. Load image data in batches
2. Extract deep visual features using a pretrained CLIP model
3. Learn a scalar **risk score** per image
4. Enforce **class separation** using pairwise and patch-level losses
5. Visualize **risk heatmaps** highlighting influential regions

---

## 🧠 High-Level Architecture

```
┌──────────────────────────┐
│      Image Dataset       │
│ (batched / streamed)    │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  CLIP Vision Backbone    │
│  (ResNet50, frozen)     │
│  Output: (B,2048,H,W)   │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   Risk Head Network      │
│  • 1×1 Conv (risk map)  │
│  • Pooling              │
│  • Small MLP            │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   Scalar Risk Logit      │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Losses & Optimization  │
│  • BCE                  │
│  • Pairwise ranking     │
│  • Patch-wise contrast  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Risk Heatmap Visualization│
└──────────────────────────┘
```

---

## 📦 Dependencies

Install required packages before running the notebook:

```bash
pip install torch timm datasets matplotlib tqdm
```

> GPU is **strongly recommended** for reasonable training speed.

---

## 📂 Notebook Structure (Cell-by-Cell)

### 1️⃣ Imports & Setup

- PyTorch, timm, datasets, matplotlib
- Device configuration (CPU / CUDA)

Purpose: **Environment initialization**

---

### 2️⃣ CLIP Feature Extractor

```python
model_name = "resnet50_clip.openai"
model = timm.create_model(
    model_name,
    pretrained=True,
    features_only=True,
    out_indices=[4]
)
```

- Uses **CLIP-pretrained ResNet50**
- Removes classifier head
- Outputs **spatial feature maps** instead of logits

Why CLIP?
- Strong semantic representations
- Better generalization under weak supervision

---

### 3️⃣ Risk Head Network

**Conceptual structure:**

```
Input: Feature map (B, 2048, H, W)

   ┌────────────────────┐
   │ 1×1 Conv (2048→1) │  → Risk Map
   └─────────┬──────────┘
             ▼
   ┌────────────────────┐
   │ Adaptive Pooling   │  → Global Risk
   └─────────┬──────────┘
             ▼
   ┌────────────────────┐n   │ MLP (1→16→1)      │  → Logit
   └────────────────────┘
```

- Produces **both**:
  - A spatial risk map (for heatmaps)
  - A scalar risk score (for loss)

---

### 4️⃣ Loss Functions

The model uses **multiple complementary losses**:

#### 🔹 Binary Cross-Entropy (BCE)

- Standard classification loss
- Operates on scalar risk logits

---

#### 🔹 Pairwise Ranking Loss

Encourages:

```
Risk(positive image) > Risk(negative image)
```

Form:

```
L = log(1 + exp(-(s_pos - s_neg)))
```

Effect:
- Improves **relative separation**
- Robust to noisy labels

---

#### 🔹 Patch-wise Loss

- Operates on **risk maps** instead of scalars
- Compares top-k spatial activations across classes
- Forces discriminative local regions

This is the key component enabling **weak localization**.

---

### 5️⃣ Training Loop

Per batch:

1. Extract CLIP features
2. Generate risk maps & logits
3. Compute all losses
4. Backpropagate
5. Update risk head parameters

CLIP backbone remains **frozen**.

---

### 6️⃣ Heatmap Visualization

Produces overlays:

```
[ Original Image ] + [ Risk Heatmap ] → Interpretable Output
```

Color intensity corresponds to **local contribution to risk**.

---

## 📊 Interpreting Results

### 🔢 Training Metrics

| Metric | Meaning |
|------|--------|
| BCE Loss | Overall classification accuracy |
| Pairwise Loss | Class separation quality |
| Patch Loss | Localization strength |
| Gap | Mean(pos) − Mean(neg) logits |

Higher **gap** = better discrimination.

---

### 🔥 Risk Heatmaps

- 🔴 Red / Yellow: high-risk regions
- 🔵 Blue / Dark: low contribution

Use cases:
- Debug spurious correlations
- Verify model reasoning
- Dataset bias detection

---

## 🧪 How to Run

1. Open `Auditor.ipynb` in Jupyter or Colab
2. Run cells **top to bottom**
3. Ensure dataset access is available
4. Monitor losses and heatmaps

---

## 🛠 Common Issues

| Problem | Solution |
|------|---------|
| CUDA OOM | Reduce batch size |
| torch not found | Run imports cell |
| No heatmap contrast | Increase patch loss weight |

---

## ❌ What This Notebook Is Not

- ❌ Not a code security auditor
- ❌ Not a repository quality checker
- ❌ Not a supervised object detector

It is a **model auditing & interpretability tool** for vision models.

---

## ✅ Summary

✔ Uses CLIP for strong representations  
✔ Learns risk under weak supervision  
✔ Produces interpretable spatial heatmaps  
✔ Lightweight and extensible

---

## 🧩 Extended Architecture (ASCII)

```
Dataset
  │
  ▼
CLIP Backbone (Frozen)
  │   Feature Maps
  ▼
Risk Conv (1×1)
  │   Spatial Risk
  ├───────────────┐
  ▼               ▼
Pooling        Heatmap Viz
  │
  ▼
MLP → Risk Logit
  │
  ▼
Losses (BCE + Pair + Patch)
  │
  ▼
Optimizer Step
```

---

## 📌 Future Extensions

- Multi-class risk heads
- Learnable attention pooling
- KTO / RLHF-style preference losses
- ViT-based CLIP backbones

---

**If you use this notebook, consider citing or linking the repository.**

