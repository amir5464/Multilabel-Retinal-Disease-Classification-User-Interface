# Eye Predictive: Retinal Disease Classification Web Interface

![Eye Predictive Dashboard](https://raw.githubusercontent.com/amir5464/Multilabel-Retinal-Disease-Classification-User-Interface/main/prediction/static/images/Screenshot%202023-12-19%20113959.png)
[![Deep Learning](https://raw.githubusercontent.com/amir5464/Multilabel-Retinal-Disease-Classification-User-Interface/main/prediction/static/images/Screenshot%202023-12-19%20114137.png)
[![Medical Imaging](https://raw.githubusercontent.com/amir5464/Multilabel-Retinal-Disease-Classification-User-Interface/main/prediction/static/images/Screenshot%202023-12-19%20121444.png)

A sophisticated web application for automated retinal disease detection and classification using deep learning.

![Eye Predictive Dashboard](https://raw.githubusercontent.com/amir5464/Multilabel-Retinal-Disease-Classification-User-Interface/main/prediction/static/images/Screenshot%202023-12-19%20113959.png)

## 🔍 Overview

Eye Predictive is a medical imaging analysis tool that uses ensemble deep learning algorithms to detect and classify multiple retinal diseases from fundus photographs. This repository contains the web interface that allows healthcare professionals to easily upload and analyze retinal images.

## ✨ Features

- **Intuitive User Interface**: Simple drag-and-drop functionality for image upload
- **Multi-disease Detection**: Identifies various retinal conditions including:
  - Optic Disc Cupping (ODC)
  - Temporal Superior Laser Photocoagulation (TSLP)
  - Macular Hole (MH)
  - And more conditions as trained in the model
- **Confidence Metrics**: Provides prediction confidence levels for each detected condition
- **Fast Processing**: Delivers results in seconds using optimized deep learning models
- **Medical-grade Results**: High accuracy, precision, and recall rates

## 🧠 Technology

The backend uses a sophisticated ensemble of deep learning models including:
- DenseNet201
- EfficientNetB3/B4/V2S
- Stacking ensemble methodology with Logistic Regression and deep neural networks

## 📊 Performance

- **Accuracy:** 93.5%
- **Precision:** 98.2%
- **Recall:** 93.9%
- **F1 Score:** 96%
- **Loss:** 0.029

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow 2.5+
- Flask 2.0+
- Other dependencies listed in `requirements.txt`

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/eye-predictive.git
   cd eye-predictive
