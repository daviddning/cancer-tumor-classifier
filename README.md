# Brain Tumor Classification - Deep Learning Model

This project uses a deep learning model based on EfficientNetV2-S transfer learning to classify MRI scans for early detection of three brain tumor types.

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
![Repo Size](https://img.shields.io/github/repo-size/daviddning/cancer-tumor-classifier?style=for-the-badge)


<img width="785" height="430" alt="image" src="https://github.com/user-attachments/assets/42fabe00-3c9c-4929-ad89-b4f51c204c86" />

## Overview

This project aims to assist medical professionals in the early and accurate diagnosis of brain tumors using deep learning. By applying deep convolutional neural networks to MRI scans, the model can detect and classify three common types of brain tumors:

- Glioma
- Meningioma
- Pituitary

## Key Features

- **Transfer Learning Architecture**: Utilized EfficientNetV2-S, pre-trained on ImageNet, and fine-tuned on a custom dataset of brain MRI scans
- **Auto Augmentation**: Applied automated data augmentation techniques to improve generalization and real-life usage  
- **Mixed Precision Training**: Leveraged mixed precision to accelerate training and reduce memory usage without compromising accuracy  

## Technologies Used

- **Python**: Primary programming language
- **PyTorch**: Deep learning framework for implementing EfficientNetV2-S
- **Jupyter**: Exploratory data analysis 
- **Kaggle Datasets**: Multiple publicly available brain tumor MRI datasets 

## Performance Metrics

| Metric        | Value   |
|--------------|---------|
| Accuracy     | 99.45%  |
| Precision    | 99.45%  |
| Recall       | 99.45%  |
| F1-Score     | 99.45%  |

## Acknowledgements

- Datasets sourced from various public repositories on Kaggle
- Inspired by recent research in AI-assisted medical imaging
