import pandas as pd
import numpy as np
import os
import joblib
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

# --- Encode labels ---
le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"\nClasses: {list(le.classes_)}")

# --- Oversample minority classes to balance dataset ---
# This duplicates underrepresented classes so the SVM doesn't ignore them
df_balanced = df.copy()
max_count = df['label'].value_counts().max()

balanced_frames = []
for label in df['label'].unique():
    df_class = df[df['label'] == label]
    df_upsampled = resample(
        df_class,
        replace=True,
        n_samples=max_count,
        random_state=42
    )
    balanced_frames.append(df_upsampled)

df_bal = pd.concat(balanced_frames)
print(f"\nAfter balancing:")
print(df_bal['label'].value_counts())

X_bal = df_bal.drop("label", axis=1).values
y_bal = df_bal["label"].values
y_bal_enc = le.transform(y_bal)

# --- Scale ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_bal)

# --- PCA ---
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
print(f"\nPCA: {X_scaled.shape[1]} → {X_pca.shape[1]} components")

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_bal_enc,
    test_size=0.2,
    random_state=42,
    stratify=y_bal_enc
)

print(f"Training: {len(X_train)}, Test: {len(X_test)}")

# --- Grid search for best SVM parameters ---
print("\nSearching for best parameters...")
param_grid = {
    'C':     [0.1, 1, 5, 10, 50],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel':['rbf', 'poly']
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(
    SVC(probability=True, random_state=42),
    param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)
grid.fit(X_train, y_train)

print(f"Best params: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_*100:.1f}%")

svm = grid.best_estimator_

# --- Evaluate on test set ---
y_pred = svm.predict(X_test)
acc = (y_pred == y_test).mean()
print(f"\n✓ Test Accuracy: {acc*100:.1f}%")

cv_scores = cross_val_score(svm, X_pca, y_bal_enc, cv=cv)
print(f"✓ 5-fold CV Accuracy: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# --- Confusion matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title("Face Shape Classifier — Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
os.makedirs("models", exist_ok=True)
plt.savefig("models/face_confusion_matrix.png", dpi=150)
plt.show()
print("✓ Confusion matrix saved.")

# --- Save models ---
joblib.dump(svm,    "models/face_shape_svm.pkl")
joblib.dump(scaler, "models/face_scaler.pkl")
joblib.dump(pca,    "models/face_pca.pkl")
joblib.dump(le,     "models/face_label_encoder.pkl")

print("\n✓ Models saved to models/")