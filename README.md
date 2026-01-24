# MedVision — Deep Learning for Medical Imaging

<div align="center">

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
![Repo Size](https://img.shields.io/github/repo-size/daviddning/cancer-tumor-classifier?style=for-the-badge)

**An end-to-end deep learning pipeline for automated brain tumor segmentation from multi-modal MRI scans**

[Features](#-features) • [Results](#-results) • [Dataset](#-dataset) • [Demo](#-demo) • [Architecture](#%EF%B8%8F-architecture)



![image](https://imgs.search.brave.com/rw3nEnp_f7jmZs3h60r1qSKXP_lmMHuql3PqRu4M9K8/rs:fit:860:0:0:0/g:ce/aHR0cHM6Ly9tZWRp/YS5nZXR0eWltYWdl/cy5jb20vaWQvMTQ0/MDUyNTQzOC9waG90/by9jbG9zZS11cC1v/Zi13b21hbi1kb2N0/b3ItbG9va2luZy1h/dC1jdC1zY2FuLXJl/cG9ydC1vbi1jb21w/dXRlci1tb25pdG9y/LmpwZz9zPTYxMng2/MTImdz0wJms9MjAm/Yz1BMmd4bjVadVA3/WFZEaF9aNk03dEJZ/TTI1NmNaaC1aTUVZ/UDN5THhpRWFJPQ)

</div>

## 🧠 Overview

Hi! I'm a second-year Data Science student at UBC, and this is my deep learning project for automated brain tumor segmentation from MRI scans.

Everything in this repository—from data exploration to model training—was built entirely on my laptop in my dorm room. The goal was to prove that meaningful medical AI research doesn’t require massive computing infrastructure, just strong fundamentals and careful engineering.

This project implements an **end-to-end segmentation pipeline** for identifying brain tumors in MRI scans using deep learning. Built on the **BraTS 2020** dataset, it demonstrates the full machine learning workflow from exploratory data analysis through model evaluation, with particular emphasis on detecting **small, clinically challenging tumors**.

## What Makes This Project Unique

- **Accessibility**  
  Trained on consumer hardware (laptop GPU) using mixed precision training and memory-efficient pipelines.

- **Clinical Focus**  
  Dedicated small tumor analysis (Dice: **0.65**), targeting regions most likely to be missed in real diagnostic settings.

- **Multi-Modal Learning**  
  Leverages four MRI modalities (**FLAIR, T1, T1ce, T2**) for comprehensive tumor characterization.

- **Research-Grade Analysis**  
  57K+ MRI slices analyzed, including tumor size distributions and slice-wise spatial analysis.

- **Complete Pipeline**  
  Everything from raw data exploration to final results, built and tested independently.


---

## 🔬 Features

### Core Capabilities
- **Attention U-Net** — Industry-standard semantic segmentation with attention gates at each decoder level
- **Mixed Precision Training** — 40% faster training with automatic mixed precision (AMP)
- **Comprehensive EDA** — Class distribution, tumor characteristics, spatial patterns analysis
- **Clinical Metrics** — Dice score, IoU, precision/recall optimized for medical imaging
- **Small Tumor Detection** — Specialized loss function with 1.5× weight for tumors <500 pixels
- **Efficient Data Pipeline** — HDF5-based loading with on-the-fly augmentation and z-score normalization

### Technical Highlights
- Patient-wise data splitting (prevents data leakage)
- Class-weighted loss functions for imbalanced segmentation
- Comprehensive logging and checkpoint management
- Reproducible experiments with fixed random seeds
- Post-training analysis with visual predictions

---

## 📈 Results

### Model Performance

| Metric | Overall |
|--------|---------|
| **Dice Score** | **0.692** |
| Precision | 0.717 | 
| Recall | 0.668 | 
| Accuracy | 99.88% | 

### Training Details
- **Training Time**: 10.7 hours (69 epochs)
- **Model Size**: 7.85M parameters
- **Best Validation Dice**: 0.743 (epoch 43)
- **Hardware**: GPU-accelerated with mixed precision

### Key Findings from EDA
- **Dataset**: 57,195 MRI slices from 369 patients
- **Class Balance**: 43% tumor vs 57% healthy (1.3:1 ratio)
- **Tumor Sizes**: 72% are small-to-medium (<2,000 pixels)
- **Spatial Pattern**: Middle brain region shows 81% tumor probability
- **Composition**: 100% necrotic tissue (simplified segmentation task)

---

## 📁 Dataset

### BraTS2020 Training Dataset
- **Source**: [Kaggle — BraTS2020 Training Data](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data)
- **Size**: 57,195 MRI slices (369 patients × 155 slices)
- **Resolution**: 240×240 pixels per slice
- **Modalities**: 4 channels (FLAIR, T1, T1ce, T2)
- **Labels**: Binary segmentation masks (0: healthy, 1: necrotic tumor)
- **Preprocessing**: Z-score normalization per channel, patient-wise data splitting

**For detailed information about preprocessing, data splits, and label definitions:**  
➡️ **[DATASETS.md](DATASETS.md)**

---

## 🚀 Demo

TEMP

## 🏗️ Architecture

### U-Net with Attention Mechanisms

```
Input (4 channels: FLAIR, T1, T1ce, T2)
    ↓
Encoder (4 levels, 32→256 filters)
    ↓
Bottleneck (512 filters)
    ↓
Decoder (4 levels with attention gates + skip connections)
    ↓
Output (1 channel: binary segmentation mask)
```

**Key architectural choices:**
- Batch normalization after each convolution
- Dropout (0.2) for regularization
- Skip connections preserve spatial information
- Attention gates at each decoder level focus on tumor regions
- Final sigmoid activation for binary segmentation

**Loss Function**: Weighted combination of three components
```python
# Enhanced Combined Loss with small tumor emphasis
loss = 0.5 × Dice_loss + 0.3 × BCE_loss + 0.2 × Focal_loss

# Where Dice loss has 1.5× weight for small tumors (<500 pixels)
# and BCE uses 1.3× weight for positive class (tumor pixels)
```

---

## 📂 Repository Structure

```
cancer-tumor-classifier/
├── graphics/eds               # Generated Visualizations
├── notebooks
│   └── eda_brain_tumor.ipynb  # Exploratory Data Analysis Notebook            
├── results
│   ├── training_loss/         # Training Loss Charts
│   └── final_results.json     # Final Test Set Metrics
├── scripts/
│   ├── config.py              # Training Configurations
│   ├── data_processing.py     # Data Preprocessing Pipeline
│   ├── losses.py              # Custom Loss Functions (Dice, Focal, etc.)
│   ├── model.py               # Attention U-Net Model Architecture
│   └── train.py               # Segmentation Training Script
├── .gitattributes             
├── .gitignore               
├── DATASETS.md                # Dataset documentation
├── LICENSE                   
└── README.md

```

---

## 🔬 Research Insights

### Small Tumor Performance
The model achieves **0.646 Dice score on tumors <500 pixels**, which is particularly impressive given:
- These tumors are often invisible to the naked eye
- Standard segmentation models struggle with small objects
- High precision required for clinical utility

### Spatial Distribution Analysis
The EDA revealed:
- **Middle brain slices (52-103)** contain 81% of tumors
- **26 slices** across all patients contain no tumors
- This spatial pattern can inform slice-wise data augmentation strategies

### Class Imbalance Handling
With a 1.3:1 healthy-to-tumor ratio:
- Applied **class weighting** in loss function
- Achieved balanced precision (0.72) and recall (0.67)
- Maintained 99.88% overall accuracy despite imbalance

---

## 🛠️ Future Improvements

- [ ] Implement 3D U-Net for volumetric segmentation
- [ ] Uncertainty quantification with Monte Carlo dropout
- [ ] Transfer learning from larger datasets (BraTS2021/2022)
- [ ] Real-time inference optimization with ONNX/TensorRT
- [ ] Web interface for clinical demonstration

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- **BraTS Challenge** for providing high-quality annotated medical imaging data
- **U-Net Architecture** from Ronneberger et al. (2015)
- Medical imaging community for established evaluation metrics

---

## 📧 Contact

**David Ning**  
[GitHub](https://github.com/daviddning) • [LinkedIn](https://www.linkedin.com/in/daviddning)

For questions about this project or collaboration opportunities, please open an issue or reach out directly.

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ for advancing medical AI research

</div>
