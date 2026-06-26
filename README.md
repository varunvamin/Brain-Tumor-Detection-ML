# Brain Tumor Detection Using Machine Learning

**Python, OpenCV, Scikit-learn, argparse, joblib**

## Overview
This project implements a brain tumor classification system using MRI images. It processes images using a complete preprocessing pipeline and classifies them to detect the presence of a brain tumor using multiple machine learning algorithms.

## Features
- **Built a brain tumor classification system** using MRI images, implementing **SVM, KNN, and Random Forest** with a complete preprocessing pipeline.
- **Applied PCA** for dimensionality reduction and evaluated models using Accuracy, Precision, Recall, and F1-Score, achieving the best accuracy with Random Forest.
- **Developed a majority voting mechanism** across all three classifiers for robust final prediction on new MRI images.
- **Modular and Robust**: The pipeline is fully refactored to support command-line execution, isolated training/prediction phases, and model persistence via `joblib`.

## Setup & Installation

1. Clone the repository and navigate into the project directory.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

This project supports two modes of execution using the command line:

### 1. Train Mode
Train the models from scratch on the dataset, visualize the results, and save the trained models (`.pkl` files) to disk.

```bash
python brain_tumor_detection.py --mode train --dataset dataset.zip
```
*(If `--dataset` is omitted, it defaults to looking for `dataset.zip` in the current folder).*

### 2. Predict Mode
Once models are trained and saved, you can predict on any new MRI image instantly without retraining.

```bash
python brain_tumor_detection.py --mode predict --image my_mri_image.jpg
```
