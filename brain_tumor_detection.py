# ============================================================
# 🧠 Brain Tumor Detection in MRI Images
# Classifiers: SVM | KNN | Random Forest | PCA
# ============================================================

# ── STEP 1: Define & Extract Dataset ─────────────────────────
import sys
import zipfile
import os

zip_name = input("Enter the path to your dataset zip file (e.g., dataset.zip) [default: dataset.zip]: ").strip()
if not zip_name:
    zip_name = "dataset.zip"

if not os.path.exists(zip_name):
    print(f"Error: '{zip_name}' not found. Please place the dataset in the current directory.")
    sys.exit(1)

print(f"Extracting: {zip_name}")

with zipfile.ZipFile(zip_name, 'r') as zip_ref:
    zip_ref.extractall('data')

print("Extracted folders:", os.listdir('data'))

# ── STEP 2: Import Libraries ──────────────────────────────────
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

print("All libraries imported successfully.")

# ── STEP 3: Load & Preprocess Images ─────────────────────────
def load_data(path):
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

            img  = cv2.resize(img, (128, 128))                 # Resize
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)       # Grayscale
            gray = cv2.GaussianBlur(gray, (5, 5), 0)           # Noise removal
            gray = gray / 255.0                                # Normalize

            data.append(gray.flatten())
            labels.append(label)

    return np.array(data), np.array(labels)


data, labels = load_data('data')

print(f"Data shape:       {data.shape}")
print(f"Labels shape:     {labels.shape}")
print(f"Tumor samples:    {np.sum(labels == 1)}")
print(f"No-tumor samples: {np.sum(labels == 0)}")

# ── STEP 4: Visualize Sample Images ──────────────────────────
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
plt.show()

# ── STEP 5: PCA ───────────────────────────────────────────────
pca = PCA(n_components=100, random_state=42)
data_pca = pca.fit_transform(data)

print(f"Shape before PCA: {data.shape}")
print(f"Shape after PCA:  {data_pca.shape}")
print(f"Variance retained: {np.sum(pca.explained_variance_ratio_) * 100:.2f}%")

# ── STEP 6: Train-Test Split ──────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    data_pca, labels, test_size=0.2, random_state=42
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# ── STEP 7: Train Classifiers ─────────────────────────────────
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)
print("SVM trained.")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print("KNN trained.")

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("Random Forest trained.")

# ── STEP 8: Accuracy Scores ───────────────────────────────────
svm_pred = svm.predict(X_test)
knn_pred = knn.predict(X_test)
rf_pred  = rf.predict(X_test)

svm_acc = accuracy_score(y_test, svm_pred)
knn_acc = accuracy_score(y_test, knn_pred)
rf_acc  = accuracy_score(y_test, rf_pred)

print("=" * 45)
print(f"  SVM Accuracy:           {svm_acc:.4f} ({svm_acc*100:.2f}%)")
print(f"  KNN Accuracy:           {knn_acc:.4f} ({knn_acc*100:.2f}%)")
print(f"  Random Forest Accuracy: {rf_acc:.4f} ({rf_acc*100:.2f}%)")
print("=" * 45)

# ── STEP 9: Classification Reports ───────────────────────────
target_names = ['No Tumor', 'Tumor']

print("\n--- SVM Classification Report ---")
print(classification_report(y_test, svm_pred, target_names=target_names))

print("\n--- KNN Classification Report ---")
print(classification_report(y_test, knn_pred, target_names=target_names))

print("\n--- Random Forest Classification Report ---")
print(classification_report(y_test, rf_pred, target_names=target_names))

# ── STEP 10: Confusion Matrices ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

for ax, pred, title in zip(axes,
                            [svm_pred, knn_pred, rf_pred],
                            ['SVM', 'KNN', 'Random Forest']):
    cm = confusion_matrix(y_test, pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f'{title} Confusion Matrix', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# ── STEP 11: Accuracy Bar Chart ───────────────────────────────
models     = ['SVM', 'KNN', 'Random Forest']
accuracies = [svm_acc, knn_acc, rf_acc]
colors     = ['steelblue', 'darkorange', 'seagreen']

plt.figure(figsize=(7, 5))
bars = plt.bar(models, accuracies, color=colors, width=0.5, edgecolor='black')

for bar, acc in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() - 0.04,
        f'{acc:.2%}',
        ha='center', va='bottom', fontsize=12, fontweight='bold', color='white'
    )

plt.ylim(0, 1.05)
plt.title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Classifier', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.tight_layout()
plt.show()

best_model = models[np.argmax(accuracies)]
print(f"\nBest performing model: {best_model} with {max(accuracies)*100:.2f}% accuracy")

# ── STEP 12: Predict on New MRI Image ────────────────────────
img_name = input("\nEnter the path to an MRI image for prediction: ").strip()

if not os.path.exists(img_name):
    print(f"Error: Image '{img_name}' not found.")
    sys.exit(1)

img  = cv2.imread(img_name)
img  = cv2.resize(img, (128, 128))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
gray = gray / 255.0

plt.imshow(gray, cmap='gray')
plt.title("Uploaded MRI Image (Preprocessed)", fontsize=12)
plt.axis('off')
plt.show()

img_flat = gray.flatten().reshape(1, -1)
img_pca  = pca.transform(img_flat)

svm_result = svm.predict(img_pca)[0]
knn_result = knn.predict(img_pca)[0]
rf_result  = rf.predict(img_pca)[0]

label_map = {1: "⚠️  TUMOR DETECTED", 0: "✅  NO TUMOR DETECTED"}

print("\n========== PREDICTION RESULTS ==========")
print(f"  SVM          : {label_map[svm_result]}")
print(f"  KNN          : {label_map[knn_result]}")
print(f"  Random Forest: {label_map[rf_result]}")
print("=========================================")

votes = [svm_result, knn_result, rf_result]
final = 1 if sum(votes) >= 2 else 0
print(f"\n  Final Decision (Majority Vote): {label_map[final]}")
