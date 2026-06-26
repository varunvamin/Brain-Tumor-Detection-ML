# Brain Tumor Detection Using Machine Learning

**Python, OpenCV, Scikit-learn, Google Colab**

## Overview
This project implements a brain tumor classification system using MRI images. It processes images using a complete preprocessing pipeline and classifies them to detect the presence of a brain tumor.

## Features
- **Built a brain tumor classification system** using MRI images, implementing **SVM, KNN, and Random Forest** with a complete preprocessing pipeline.
- **Applied PCA** for dimensionality reduction and evaluated models using Accuracy, Precision, Recall, and F1-Score, achieving the best accuracy with Random Forest.
- **Developed a majority voting mechanism** across all three classifiers for robust final prediction on new MRI images.

## Setup & Usage (Google Colab)
This project was primarily developed to run in **Google Colab**.
To test it, upload the `brain_tumor_detection.py` script contents into a Google Colab notebook cell and execute. 
It utilizes `google.colab.files` for uploading datasets and individual test images directly in the browser.
