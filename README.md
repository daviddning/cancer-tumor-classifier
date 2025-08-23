# Brain Tumor Classification - Deep Learning Model

This project uses a deep learning model based on EfficientNetV2-S transfer learning to classify MRI scans for early detection of three brain tumor types.

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
![Repo Size](https://img.shields.io/github/repo-size/daviddning/cancer-tumor-classifier?style=for-the-badge)


<img width="860" height="507" alt="image" src="https://github.com/user-attachments/assets/42fabe00-3c9c-4929-ad89-b4f51c204c86" />

## Project Overview

This project aims to assist medical professionals in the early and accurate diagnosis of brain tumors using artificial intelligence. By applying deep convolutional neural networks to MRI scans, the model can detect and classify three common types of brain tumors:

- Glioma
- Meningioma
- Pituitary

## Key Features

- **Transfer Learning Architecture**: Utilized EfficientNetV2-S, pre-trained on ImageNet, and fine-tuned on a custom dataset of brain MRI scans
- **High-Quality MRI Dataset**: Aggregated from four Kaggle datasets, providing a diverse and representative set of brain tumor images
- **Multi-Class Classification**: Accurately classifies images into three brain tumor categories
- **High Accuracy**: Achieved a test accuracy of 99.45% during training and validation
- **Robust Evaluation**: Validated on an independent foreign dataset to ensure real-world applicability

## Applications

- **Hospital Diagnosis Assistance**: Can be integrated into hospital systems to support radiologists
- **Telemedicine & Remote Health**: Offers diagnostic support in areas with limited access to specialists
- **AI-Powered Radiology Tools**: Serves as a foundational model for future AI-driven diagnostic tools

## Technologies Used

- **Python**: Primary programming language
- **PyTorch**: Deep learning framework for implementing EfficientNetV2-S
- **Kaggle Datasets**: Multiple publicly available brain tumor MRI datasets

## Performance Metrics


| Metric        | Value   |
|--------------|---------|
| Accuracy     | 99.45%  |
| Precision    | 99.2%   |
| Recall       | 99.3%   |
| F1-Score     | 99.25%  |


## Acknowledgements

- Datasets sourced from various public repositories on Kaggle
- Inspired by recent research in AI-assisted medical imaging
