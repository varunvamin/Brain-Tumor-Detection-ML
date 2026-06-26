# ============================================================
# 🧠 Brain Tumor Detection in MRI Images
# Classifiers: SVM | KNN | Random Forest | PCA
# ============================================================

import argparse
import sys
import zipfile
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

def extract_dataset(zip_path, extract_to='data'):
    """Extracts the dataset zip file."""
    if not os.path.exists(zip_path):
        print(f"Error: '{zip_path}' not found.")
        sys.exit(1)
    
    if not os.path.exists(extract_to):
        print(f"Extracting: {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete.")
    else:
        print(f"Dataset already extracted to '{extract_to}'.")

def load_data(path):
    """Loads and preprocesses images from the dataset folder."""
    data = []
    labels = []

    for category in ['yes', 'no']:
        folder = os.path.join(path, category)
        if not os.path.exists(folder):
            print(f"Warning: folder '{folder}' not found, skipping.")
            continue

        label = 1 if category == 'yes' else 0  # 1 = Tumor, 0 = No Tumor

        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Preprocessing
            img  = cv2.resize(img, (128, 128))                 # Resize
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)       # Grayscale
            gray = cv2.GaussianBlur(gray, (5, 5), 0)           # Noise removal
            gray = gray / 255.0                                # Normalize

            data.append(gray.flatten())
            labels.append(label)

    return np.array(data), np.array(labels)

def train_models(dataset_zip):
    """Trains the models from scratch, visualizes results, and saves them."""
    extract_dataset(dataset_zip, 'data')
    
    print("\nLoading and preprocessing images...")
    data, labels = load_data('data')

    if len(data) == 0:
        print("Error: No images found in dataset. Make sure the folders 'yes' and 'no' exist inside 'data/'.")
        sys.exit(1)

    print(f"Data shape:       {data.shape}")
    print(f"Tumor samples:    {np.sum(labels == 1)}")
    print(f"No-tumor samples: {np.sum(labels == 0)}")

    # Visualize Sample Images
    tumor_idx    = np.where(labels == 1)[0][:2]
    no_tumor_idx = np.where(labels == 0)[0][:2]
    sample_indices = list(tumor_idx) + list(no_tumor_idx)
    titles = ['Tumor', 'Tumor', 'No Tumor', 'No Tumor']

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, idx, title in zip(axes, sample_indices, titles):
        ax.imshow(data[idx].reshape(128, 128), cmap='gray')
        ax.set_title(title, fontsize=12)
        ax.axis('off')

    plt.suptitle('Sample MRI Images After Preprocessing', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    print("\nApplying PCA...")
    pca = PCA(n_components=100, random_state=42)
    data_pca = pca.fit_transform(data)
    joblib.dump(pca, 'pca_model.pkl')
    print(f"Variance retained: {np.sum(pca.explained_variance_ratio_) * 100:.2f}%")

    X_train, X_test, y_train, y_test = train_test_split(data_pca, labels, test_size=0.2, random_state=42)

    print("\nTraining Classifiers...")
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train, y_train)
    joblib.dump(svm, 'svm_model.pkl')
    print("SVM trained.")

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    joblib.dump(knn, 'knn_model.pkl')
    print("KNN trained.")

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    joblib.dump(rf, 'rf_model.pkl')
    print("Random Forest trained.")

    print("\nSaving Models to disk (.pkl files)... Done.")

    # Evaluations
    svm_pred = svm.predict(X_test)
    knn_pred = knn.predict(X_test)
    rf_pred  = rf.predict(X_test)

    svm_acc = accuracy_score(y_test, svm_pred)
    knn_acc = accuracy_score(y_test, knn_pred)
    rf_acc  = accuracy_score(y_test, rf_pred)

    print("\n" + "=" * 45)
    print(f"  SVM Accuracy:           {svm_acc:.4f} ({svm_acc*100:.2f}%)")
    print(f"  KNN Accuracy:           {knn_acc:.4f} ({knn_acc*100:.2f}%)")
    print(f"  Random Forest Accuracy: {rf_acc:.4f} ({rf_acc*100:.2f}%)")
    print("=" * 45)

    target_names = ['No Tumor', 'Tumor']

    # Confusion Matrices Visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, pred, title in zip(axes, [svm_pred, knn_pred, rf_pred], ['SVM', 'KNN', 'Random Forest']):
        cm = confusion_matrix(y_test, pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(f'{title} Confusion Matrix', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    # Accuracy Bar Chart Visualization
    models = ['SVM', 'KNN', 'Random Forest']
    accuracies = [svm_acc, knn_acc, rf_acc]
    plt.figure(figsize=(7, 5))
    bars = plt.bar(models, accuracies, color=['steelblue', 'darkorange', 'seagreen'], width=0.5, edgecolor='black')

    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.04, f'{acc:.2%}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold', color='white')

    plt.ylim(0, 1.05)
    plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Classifier', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.tight_layout()
    plt.show()

def predict_image(img_path):
    """Loads saved models and predicts the class of a new image."""
    if not os.path.exists(img_path):
        print(f"Error: Image '{img_path}' not found.")
        sys.exit(1)

    required_models = ['pca_model.pkl', 'svm_model.pkl', 'knn_model.pkl', 'rf_model.pkl']
    for model_file in required_models:
        if not os.path.exists(model_file):
            print(f"Error: '{model_file}' not found. Please train the models first using --mode train.")
            sys.exit(1)

    # Load models
    pca = joblib.load('pca_model.pkl')
    svm = joblib.load('svm_model.pkl')
    knn = joblib.load('knn_model.pkl')
    rf  = joblib.load('rf_model.pkl')

    # Load and preprocess image
    img  = cv2.imread(img_path)
    img  = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = gray / 255.0

    img_flat = gray.flatten().reshape(1, -1)
    img_pca  = pca.transform(img_flat)

    svm_result = svm.predict(img_pca)[0]
    knn_result = knn.predict(img_pca)[0]
    rf_result  = rf.predict(img_pca)[0]

    label_map = {1: "⚠️ TUMOR DETECTED", 0: "✅ NO TUMOR DETECTED"}

    print("\n========== PREDICTION RESULTS ==========")
    print(f"  SVM          : {label_map[svm_result]}")
    print(f"  KNN          : {label_map[knn_result]}")
    print(f"  Random Forest: {label_map[rf_result]}")
    print("=========================================")

    votes = [svm_result, knn_result, rf_result]
    final = 1 if sum(votes) >= 2 else 0
    print(f"\n  Final Decision (Majority Vote): {label_map[final]}")

    plt.imshow(gray, cmap='gray')
    plt.title(f"Prediction: {label_map[final]}", fontsize=12)
    plt.axis('off')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Brain Tumor Detection using MRI Images")
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], required=True, 
                        help="Mode: 'train' to train models and evaluate, 'predict' to test a specific image")
    parser.add_argument('--dataset', type=str, default="dataset.zip", 
                        help="Path to the dataset zip file (used in train mode)")
    parser.add_argument('--image', type=str, 
                        help="Path to the MRI image (used in predict mode)")

    args = parser.parse_args()

    if args.mode == 'train':
        train_models(args.dataset)
    elif args.mode == 'predict':
        if not args.image:
            print("Error: The --image argument is required when mode is 'predict'.")
            sys.exit(1)
        predict_image(args.image)

if __name__ == "__main__":
    main()
