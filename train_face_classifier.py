import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use("Agg")  # non-blocking backend, no window pops up
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample

# --- Load features ---
df = pd.read_csv("data/face_features.csv")
print(f"Dataset: {len(df)} samples across {df['label'].nunique()} classes")
print(df['label'].value_counts())

X = df.drop("label", axis=1).values
y = df["label"].values

le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"\nClasses: {list(le.classes_)}  ({X.shape[1]} features)")

# --- Split FIRST (on real data, before any oversampling) ---
# This prevents duplicated rows from leaking across train/test.
X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# --- Oversample the TRAINING set only ---
train_df = pd.DataFrame(X_train_raw)
train_df["label"] = y_train_raw
max_count = train_df["label"].value_counts().max()
frames = []
for label in train_df["label"].unique():
    cls = train_df[train_df["label"] == label]
    up = resample(cls, replace=True, n_samples=max_count, random_state=42)
    frames.append(up)
train_bal = pd.concat(frames)
X_train_bal = train_bal.drop("label", axis=1).values
y_train_bal = train_bal["label"].values
print(f"\nTrain after balancing: {len(X_train_bal)}  |  Test (untouched): {len(X_test)}")

# --- Scale (fit on train only) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled  = scaler.transform(X_test)

# --- PCA (fit on train only) ---
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca  = pca.transform(X_test_scaled)
print(f"PCA: {X_train_scaled.shape[1]} -> {X_train_pca.shape[1]} components")

# --- Grid search ---
print("\nSearching for best parameters...")
param_grid = {
    'C':     [0.1, 1, 5, 10, 50, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel':['rbf', 'poly']
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    SVC(probability=True, random_state=42),
    param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0
)
grid.fit(X_train_pca, y_train_bal)
print(f"Best params: {grid.best_params_}")
print(f"Best CV score (train): {grid.best_score_*100:.1f}%")
svm = grid.best_estimator_

# --- Evaluate on the held-out, non-oversampled test set ---
y_pred = svm.predict(X_test_pca)
acc = (y_pred == y_test).mean()
print(f"\nTest Accuracy (honest, no leakage): {acc*100:.1f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# --- Confusion matrix (saved, not shown) ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Face Shape Classifier - Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
os.makedirs("models", exist_ok=True)
plt.savefig("models/face_confusion_matrix.png", dpi=150)
plt.close()
print("Confusion matrix saved to models/face_confusion_matrix.png")

# --- Save models ---
joblib.dump(svm,    "models/face_shape_svm.pkl")
joblib.dump(scaler, "models/face_scaler.pkl")
joblib.dump(pca,    "models/face_pca.pkl")
joblib.dump(le,     "models/face_label_encoder.pkl")
print("\nModels saved to models/")