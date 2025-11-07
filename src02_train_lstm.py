import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import itertools

CLASSES = ["jab", "hook", "uppercut"]

# ---------- Load ----------
X = np.load("extracts/X.npy")       # (N, 30, 17, 3)
y = np.load("extracts/y.npy")       # (N,)

# (optional) add velocities (frame-to-frame deltas) to improve temporal cues
def add_velocity(X):
    vel = np.diff(X[...,:2], axis=1, prepend=X[:, :1, :,:2])
    # concat velocity (x,y) with conf channel
    conf = X[..., 2:3]
    XY = X[...,:2]
    X_feat = np.concatenate([XY, conf, vel], axis=-1)  # shape: (N, T, 17, 5)
    return X_feat

X = add_velocity(X)                 # (N, 30, 17, 5)

# flatten keypoints per frame: 17 * 5 = 85 features
N, T, J, C = X.shape
X = X.reshape(N, T, J*C).astype("float32")

# ---------- Split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# class weights (help if one class has fewer samples)
classes, counts = np.unique(y_train, return_counts=True)
max_c = counts.max()
class_weight = {int(c): float(max_c/n) for c, n in zip(classes, counts)}

# ---------- Model ----------
model = models.Sequential([
    layers.Input(shape=(T, J*C)),       # (30, 85)
    layers.Masking(mask_value=0.0),
    layers.LSTM(128, return_sequences=True),
    layers.LSTM(128),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.35),
    layers.Dense(len(CLASSES), activation="softmax")
])

model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

ckpt = callbacks.ModelCheckpoint(
    "models/punch_lstm_best.h5", monitor="val_accuracy",
    save_best_only=True, verbose=1
)
es = callbacks.EarlyStopping(
    monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1
)

hist = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=40,
    batch_size=64,
    class_weight=class_weight,
    callbacks=[ckpt, es],
    verbose=1
)

# ---------- Evaluate ----------
pred = model.predict(X_test, verbose=0).argmax(axis=1)
print(classification_report(y_test, pred, target_names=CLASSES))

# Simple confusion-matrix print (text)
cm = confusion_matrix(y_test, pred)
print("Confusion matrix:\n", cm)

# ---------- Save ----------
model.save("models/punch_lstm.h5")
print("Saved final model to models/punch_lstm.h5")
