
# ArtExtract – GSoC Evaluation Tasks

This repository contains my solution for the **ArtExtract Project (HumanAI @ CERN)** GSoC evaluation tasks.

---

## Overview

The goal is to:
- Build models for **Style, Genre, Artist classification**
- Learn meaningful **visual representations (embeddings)**
- Detect **outliers in artworks**
- Implement **similarity search**

---

## Approach

### Task 1 – Classification + Representation Learning

- Used **EfficientNetV2B2** as backbone
- Designed **task-specific architectures**:
  - **Style** → GeM pooling (texture-focused)
  - **Genre** → GeM + Global Average Pooling (semantic)
  - **Artist** → Attention + GeM (fine-grained features)

- Training strategy:
  - Phase 1 → frozen backbone
  - Phase 2 → partial fine-tuning
  - Phase 3 → controlled deep tuning

---

### Task 1 – Outlier Detection

- Extracted embeddings from trained models
- Computed **class centroids**
- Measured **Euclidean distance from centroid**
- Identified:
  - paintings far from cluster center
  - potential label noise / ambiguous styles

---

### Task 2 – Similarity Search

- Used learned embeddings
- Applied **cosine similarity**
- Retrieved visually similar artworks

---

## Results

### Classification Performance

| Task   | Accuracy |
|--------|--------|
| Style  | ~30-32% |
| Genre  | ~30–37% |
| Artist | ~12–26% |

---

## Embedding Analysis

### Distance Distribution

- Most samples cluster near centroid
- Long-tail distribution indicates meaningful outliers

<img width="679" height="509" alt="image" src="https://github.com/user-attachments/assets/d6562592-eb67-4f41-9380-73001bdb396b" />
<img width="713" height="518" alt="image" src="https://github.com/user-attachments/assets/149bd873-fe5e-4980-befd-a86b4e322e0d" />

---

### UMAP Visualization

- Continuous manifolds observed
- Partial clustering with overlap (expected for art styles)

<img width="604" height="593" alt="image" src="https://github.com/user-attachments/assets/4811d044-dcfc-4dcd-9b22-9d416695e39f" />

---

## Outlier Examples

- Some artworks deviate significantly from their assigned class
- Observed:
  - monochrome sketches inside colorful style clusters
  - stylistic deviations within same artist

<img width="616" height="354" alt="image" src="https://github.com/user-attachments/assets/7d0e9279-6dae-4869-9c1d-fec76b80e834" />
<img width="605" height="424" alt="image" src="https://github.com/user-attachments/assets/f01cebb7-375d-413f-bdee-4bc59cf88f7a" />

---

## Similarity Search

- Retrieved visually similar artworks using embeddings
- Demonstrates learned representation captures:
  - color composition
  - structure
  - style

<img width="548" height="456" alt="image" src="https://github.com/user-attachments/assets/e63c7e18-adc8-41e8-81e6-227f2a717165" />
<img width="602" height="433" alt="image" src="https://github.com/user-attachments/assets/f40bf598-fa9b-41a6-b71d-aef8f9bee422" />

---

## Key Insights

- Style classification relies heavily on **texture patterns**
- Genre requires **semantic understanding**
- Artist classification is hardest due to:
  - high intra-class variance
- Embeddings capture meaningful relationships beyond labels

---

## Tech Stack

- TensorFlow / Keras
- EfficientNetV2
- UMAP
- NumPy / OpenCV / Matplotlib

---

## Files

- `notebook.ipynb` → full implementation
- `report.pdf` → detailed explanation
- `images/` → visual outputs

---

## Conclusion

This work demonstrates:
- effective representation learning
- meaningful embedding space
- interpretability via outlier detection

---
