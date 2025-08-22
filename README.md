# Brain Tumor Classification - Deep Learning Model

![Brain Tumor Classification](https://via.placeholder.com/800x400?text=Brain+Tumor+Classification) 

This project presents a deep learning-based model for the early detection of brain tumors using magnetic resonance imaging (MRI) data. The model leverages transfer learning with the EfficientNetV2-S architecture to classify MRI images into one of three tumor types.

## Project Overview

This project aims to assist medical professionals in the early and accurate diagnosis of brain tumors using artificial intelligence. By applying deep convolutional neural networks to MRI scans, the model can detect and classify three common types of brain tumors:

- Glioma
- Meningioma
- Pituitary

Using transfer learning with EfficientNetV2-S, a state-of-the-art CNN architecture, the model was trained on a large and diverse dataset compiled from multiple Kaggle sources, ensuring strong generalization capabilities.

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
