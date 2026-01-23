# Datasets

This project uses publicly available brain MRI datasets for training and evaluating deep learning models for **automated brain tumor segmentation**.  
All data is used strictly for **educational and research purposes**.

---

## 1. BraTS 2020 — Brain Tumor Segmentation Dataset

### Overview
The primary dataset used in this project is the **Brain Tumor Segmentation (BraTS) 2020** training dataset. BraTS is a widely used benchmark in medical imaging research and provides multi-modal MRI scans with expert-annotated tumor masks.

- **Dataset Name:** BraTS 2020 Training Dataset  
- **Source:** Kaggle  
- **Link:** https://www.kaggle.com/datasets/awsaf49/brats2020-training-data  
- **Challenge Website:** https://www.med.upenn.edu/cbica/brats2020/

---

### Dataset Composition

- **Patients:** 369 glioma patients  
- **Slices:** 57,195 axial MRI slices  
- **Slices per patient:** 155 (fixed)  
- **Image resolution:** 240 × 240 pixels  
- **Modalities (4 channels):**
  - FLAIR
  - T1
  - T1ce (contrast-enhanced)
  - T2

Each slice is stored in **HDF5 (`.h5`) format**, containing:
- `image`: 4-channel MRI input
- `mask`: Binary segmentation mask

---

### Labels & Segmentation Task

This project focuses on **binary tumor segmentation**.

- **Label encoding:**
  - `0` — Healthy tissue
  - `1` — Necrotic tumor tissue

---

### Class Distribution (Slice-Level)

- **Tumor slices:** 24,369 (43%)
- **Healthy slices:** 32,826 (57%)
- **Class ratio:** ~1.3:1 (healthy:tumor)

All patients contain tumors; “healthy” slices correspond to **tumor-free regions within glioma patients**, not scans from healthy individuals.

---

### Tumor Size Characteristics

Based on sampled tumor slices:

- **Tumor size range:** 1 – 6,293 pixels
- **Mean size:** 1,488 pixels
- **Median size:** 1,253 pixels
- **Small-to-medium tumors (<2,000 px):** ~72% of samples

These characteristics motivated the use of:
- Attention mechanisms
- Small-tumor–weighted loss functions

---

### Spatial Distribution

Tumor occurrence varies by slice position:

- **Middle brain region (slices 52–103):**
  - ~81% tumor probability
- **Slices with no tumors:** 26 total
- **Top and bottom regions:** Moderate tumor presence

This spatial bias informed slice-wise analysis and augmentation strategies.

---

### Metadata Files

The dataset includes accompanying CSV metadata files used for analysis and preprocessing:

- **Slice-level metadata**
  - Tumor presence (`target`)
  - Slice index
  - Patient / volume ID
  - Tumor pixel counts
- **Clinical metadata (optional)**
  - Survival data and demographics  
  - *Not used for modeling in this project*

---

### Preprocessing

The following preprocessing steps are applied:

- Z-score normalization per MRI channel
- Patient-wise train/validation/test splitting (prevents data leakage)
- On-the-fly data augmentation during training
- Efficient HDF5-based loading for memory-constrained environments

---

### Ethical & Usage Notes

- All data is **publicly available**
- No personally identifiable information is included
- Intended for **research and educational use only**
- Not approved for clinical deployment

---

## References

- Menze et al., *The Multimodal Brain Tumor Image Segmentation Benchmark (BraTS)*  
- Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*

---

If you use this dataset or repository, please cite the original BraTS authors.
