# src/02_train_lstm.py
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ---------------- Config ----------------
X_PATH = "extracts/X.npy"   # shape (N, T=30, 17, 3) -> [x_norm, y_norm, conf]
Y_PATH = "extracts/y.npy"
MODEL_DIR = "models"
MODEL_BEST = os.path.join(MODEL_DIR, "punch_lstm_best.keras")
MODEL_FINAL = os.path.join(MODEL_DIR, "punch_lstm.keras")
CLASS_NAMES = ["jab", "hook", "uppercut"]

os.makedirs(MODEL_DIR, exist_ok=True)

# --------------- Load -------------------
print("[INFO] Loading data...")
X = np.load(X_PATH).astype("float32")  # already shoulder-normalized per frame
y = np.load(Y_PATH).astype("int64")
N, T, J, C = X.shape
print(f"[INFO] X: {X.shape}, y: {y.shape} (classes: {dict(zip(*np.unique(y, return_counts=True)) )})")

# ------- Add velocity features (key fix) -------
# Δx, Δy across time; keep conf channel as-is
vel = np.diff(X[..., :2], axis=1, prepend=X[:, :1, :, :2])  # (N,T,17,2)
X_feat = np.concatenate([X[..., :2], X[..., 2:3], vel], axis=-1)  # (N,T,17,5)
F = J * 5  # 85 features per frame
X_feat = X_feat.reshape(N, T, F).astype("float32")

# Optional standardization per-feature (helps stability)
# Compute on train split only
X_train, X_test, y_train, y_test = train_test_split(
    X_feat, y, test_size=0.2, stratify=y, random_state=42
)
mu = X_train.mean(axis=(0, 1), keepdims=True)
sigma = X_train.std(axis=(0, 1), keepdims=True) + 1e-6
X_train = (X_train - mu) / sigma
X_test  = (X_test  - mu) / sigma

# Class weights to discourage single-class collapse
classes, counts = np.unique(y_train, return_counts=True)
max_c = counts.max()
class_weight = {int(c): float(max_c / n) for c, n in zip(classes, counts)}
print("[INFO] class_weight:", class_weight)

# --------------- Model -------------------
tf.keras.utils.set_random_seed(42)

model = models.Sequential([
    layers.Input(shape=(T, F)),           # (30, 85)
    layers.Masking(mask_value=0.0),
    layers.LSTM(128, return_sequences=True),
    layers.Dropout(0.35),
    layers.LSTM(128),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.35),
    layers.Dense(len(CLASS_NAMES), activation="softmax"),
])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

ckpt = callbacks.ModelCheckpoint(MODEL_BEST, monitor="val_accuracy",
                                 save_best_only=True, verbose=1)
es   = callbacks.EarlyStopping(monitor="val_accuracy",
                               patience=6, restore_best_weights=True, verbose=1)

print("[INFO] Training...")
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=40,
    batch_size=64,
    class_weight=class_weight,
    callbacks=[ckpt, es],
    verbose=1
)

# --------------- Evaluate ----------------
probs = model.predict(X_test, verbose=0)
y_pred = probs.argmax(axis=1)

print("\n[REPORT]")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# --------------- Save --------------------
model.save(MODEL_FINAL)
print(f"[INFO] Saved final model -> {MODEL_FINAL}")
print(f"[INFO] Best checkpoint -> {MODEL_BEST}")
