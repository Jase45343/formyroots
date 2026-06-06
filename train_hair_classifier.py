import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample

# --- Load ---
df = pd.read_csv("data/hair_features.csv")
print(f"Dataset: {len(df)} samples across {df['label'].nunique()} classes")
print(df['label'].value_counts())

X = df.drop("label", axis=1).values
y = df["label"].values

# --- Encode ---
le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"\nClasses: {list(le.classes_)}")

# --- Balance classes ---
max_count = df['label'].value_counts().max()
balanced_frames = []
for label in df['label'].unique():
    df_class = df[df['label'] == label]
    df_up = resample(df_class, replace=True,
                     n_samples=max_count, random_state=42)
    balanced_frames.append(df_up)

df_bal = pd.concat(balanced_frames)
print(f"\nAfter balancing: {df_bal['label'].value_counts().to_dict()}")

X_bal = df_bal.drop("label", axis=1).values
y_bal_enc = le.transform(df_bal["label"].values)

# --- Scale ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_bal)

# --- PCA ---
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA: {X_scaled.shape[1]} → {X_pca.shape[1]} components")

# --- Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y_bal_enc, test_size=0.2,
    random_state=42, stratify=y_bal_enc
)

# --- Try both SVM and Random Forest, keep the best ---
print("\nTraining SVM...")
svm_params = {'C': [1, 10, 50], 'gamma': ['scale', 'auto'], 'kernel': ['rbf']}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
svm_grid = GridSearchCV(SVC(probability=True, random_state=42),
                        svm_params, cv=cv, scoring='accuracy', n_jobs=-1)
svm_grid.fit(X_train, y_train)

print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=None,
    random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)

# --- Compare ---
svm_acc  = svm_grid.best_estimator_.score(X_test, y_test)
rf_acc   = rf.score(X_test, y_test)

print(f"\nSVM Test Accuracy:          {svm_acc*100:.1f}%")
print(f"Random Forest Test Accuracy: {rf_acc*100:.1f}%")

# Pick the better model
if svm_acc >= rf_acc:
    best_model = svm_grid.best_estimator_
    model_name = "SVM"
else:
    best_model = rf
    model_name = "Random Forest"

print(f"\n✓ Using: {model_name}")

y_pred = best_model.predict(X_test)
acc = (y_pred == y_test).mean()

cv_scores = cross_val_score(best_model, X_pca, y_bal_enc, cv=cv)
print(f"✓ Test Accuracy:    {acc*100:.1f}%")
print(f"✓ 5-fold CV Accuracy: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# --- Confusion matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title(f"Hair Type Classifier ({model_name}) — Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
os.makedirs("models", exist_ok=True)
plt.savefig("models/hair_confusion_matrix.png", dpi=150)
plt.show()
print("✓ Confusion matrix saved to models/hair_confusion_matrix.png")

# --- Save ---
joblib.dump(best_model, "models/hair_type_model.pkl")
joblib.dump(scaler,     "models/hair_scaler.pkl")
joblib.dump(pca,        "models/hair_pca.pkl")
joblib.dump(le,         "models/hair_label_encoder.pkl")

print(f"\n✓ Models saved to models/")