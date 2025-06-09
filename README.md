# Brain Tumor Clasification - Deep Learning Model

This project presents a deep learning-based model for the early detection of brain tumors using magnetic resonance imaging (MRI) data. The model leverages transfer learning with the EfficientNetV2-S architecture to classify MRI images into one of three tumor types: glioma, meningioma, or pituitary. The model achieved an impressive 99.45% test accuracy and was validated on an external dataset for robustness.

## Project Overview
This project aims to assist medical professionals in the early and accurate diagnosis of brain tumors using artificial intelligence. By applying deep convolutional neural networks to MRI scans, the model can detect and classify three common types of brain tumors:
- Glioma
- Meningioma
- Pituitary

Using transfer learning with EfficientNetV2-S , a state-of-the-art CNN architecture, the model was trained on a large and diverse dataset compiled from multiple Kaggle sources, ensuring strong generalization capabilities.

## Key Features
Transfer Learning Architecture: 
Utilized EfficientNetV2-S , pre-trained on ImageNet, and fine-tuned it on a custom dataset of brain MRI scans.
High-Quality MRI Dataset: 
Aggregated from four Kaggle datasets , providing a diverse and representative set of brain tumor images.
Multi-Class Classification: 
Accurately classifies images into three brain tumor categories .
High Accuracy: 
Achieved a test accuracy of 99.45% during training and validation, with strong performance on an external dataset.
Robust Evaluation: 
Validated on an independent foreign dataset to ensure real-world applicability and generalizability.

## Applications
Hospital Diagnosis Assistance: 
Can be integrated into hospital systems to support radiologists in making faster and more accurate brain tumor diagnoses.
Telemedicine & Remote Health: 
Offers diagnostic support in areas with limited access to specialized neuroimaging experts.
AI-Powered Radiology Tools: 
Serves as a foundational model for future AI-driven tools in medical imaging and diagnostics.

## Technologies Used
Python: 
Primary programming language for data processing, modeling, and evaluation.
PyTorch: 
Deep learning framework used to implement transfer learning with EfficientNetV2-S.
Kaggle Datasets: 
Multiple publicly available brain tumor MRI image datasets were combined for training.

## Performance Metrics


## Acknowledgements
- Datasets sourced from various public repositories on Kaggle
- Inspired by recent research in AI-assisted medical imaging and computer vision applications in healthcare.













