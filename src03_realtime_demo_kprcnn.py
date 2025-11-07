# src/03_realtime_demo_kprcnn.py
import os, time, csv
from collections import deque, defaultdict

import cv2, numpy as np, torch, tensorflow as tf
from torchvision import transforms
from torchvision.models.detection import keypointrcnn_resnet50_fpn

# -------------------- Config --------------------
CLASSES = ["jab", "hook", "uppercut"]
SEQ_LEN = 30
SMOOTH = 0.35                 # EMA smoothing weight for new probs
COOLDOWN = 8                  # frames between counting same class
THRESHES = {"jab": 0.55, "hook": 0.70, "uppercut": 0.55}
MIN_WRIST_SPEED = 0.015       # motion gate (normalized)
TEXT = (0,255,0); ACCENT = (0,165,255)

# LSTM model(s)
MODEL_CANDIDATES = [
    "models/punch_lstm_best.keras",
    "models/punch_lstm.keras",
    "models/punch_lstm_best.h5",
    "models/punch_lstm.h5",
]

# Optional standardization (if you saved mu/sigma in training)
MU_PATH = "models/mu.npy"
SIGMA_PATH = "models/sigma.npy"

# Keypoint R-CNN local weights (no download)
LOCAL_WEIGHTS = r"models\weights\keypointrcnn_resnet50_fpn_coco.pth"

# COCO keypoint indices we rely on
L_WRIST, R_WRIST = 9, 10
L_SHOULDER, R_SHOULDER = 5, 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
to_tensor = transforms.ToTensor()
# ------------------------------------------------


def load_kprcnn_local():
    if not os.path.exists(LOCAL_WEIGHTS):
        raise FileNotFoundError(
            f"Missing weights at {LOCAL_WEIGHTS}. "
            "Place the COCO file there (keypointrcnn_resnet50_fpn_coco-*.pth)."
        )
    print(f"[INFO] Loading Keypoint R-CNN from {LOCAL_WEIGHTS}")
    m = keypointrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    state = torch.load(LOCAL_WEIGHTS, map_location=device)
    m.load_state_dict(state)
    return m.to(device).eval()


def load_classifier():
    for p in MODEL_CANDIDATES:
        if os.path.exists(p):
            print(f"[INFO] Loading LSTM classifier: {p}")
            m = tf.keras.models.load_model(p)
            # compile to silence TF warning when predicting
            m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            return m
    raise FileNotFoundError("No trained LSTM model found in models/")

def load_mu_sigma():
    if os.path.exists(MU_PATH) and os.path.exists(SIGMA_PATH):
        mu = np.load(MU_PATH); sigma = np.load(SIGMA_PATH)
        print("[INFO] Loaded mu/sigma for standardization.")
        return mu, sigma
    print("[WARN] mu/sigma not found. Running without standardization (should still work).")
    return None, None

@torch.inference_mode()
def detect_kpts_bgr(frame_bgr, model):
    """Return (17,3) [x,y,conf] for the largest-person detection, or None."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inp = to_tensor(rgb).to(device)   # (3,H,W) float32 [0..1]
    out = model([inp])[0]
    if len(out["keypoints"]) == 0:
        return None
    scores = out["scores"].detach().cpu().numpy()
    idx = int(scores.argmax())
    kpts = out["keypoints"][idx].detach().cpu().numpy()  # (17,3) [x,y,vis]
    conf = np.clip(scores[idx], 0.0, 1.0)
    kpts = np.concatenate([kpts[:, :2], np.full((17,1), conf, dtype=np.float32)], axis=1)
    return kpts

def normalize_coco17(kpts):
    """Center on shoulder midpoint; scale by shoulder distance; keep conf."""
    left, right = kpts[L_SHOULDER,:2], kpts[R_SHOULDER,:2]
    center = (left + right) / 2.0
    scale  = float(np.linalg.norm(left - right) + 1e-6)
    xy   = (kpts[:, :2] - center) / scale
    conf = kpts[:, 2:3]
    return np.concatenate([xy, conf], axis=1)  # (17,3)

def add_velocity(seq_173):
    """(T,17,3) -> concat vel (dx,dy) -> (T,17,5)"""
    vel = np.diff(seq_173[..., :2], axis=0, prepend=seq_173[:1, ..., :2])
    conf = seq_173[..., 2:3]
    xy = seq_173[..., :2]
    return np.concatenate([xy, conf, vel], axis=-1)  # (T,17,5)

def avg_wrist_speed(seq_173):
    wrists = seq_173[:, [L_WRIST, R_WRIST], :2]   # (T,2,2)
    dv = np.diff(wrists, axis=0, prepend=wrists[:1])
    return float(np.linalg.norm(dv, axis=-1).mean())

def draw_prob_bars(frame, probs, x=20, y=140, w=220, h=18, gap=8):
    if probs is None: return
    y0 = y
    for i, c in enumerate(CLASSES):
        p = float(probs[i]); pw = int(w * p)
        cv2.rectangle(frame, (x, y0), (x+w, y0+h), (60,60,60), 1)
        cv2.rectangle(frame, (x, y0), (x+pw, y0+h), (80,220,80), -1)
        cv2.putText(frame, f"{c}: {p*100:4.1f}%", (x+w+10, y0+h-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT, 2, cv2.LINE_AA)
        y0 += h + gap


def main():
    # Models
    kprcnn = load_kprcnn_local()
    clf = load_classifier()
    mu, sigma = load_mu_sigma()  # optional

    # Camera (Windows DirectShow; fallbacks)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[WARN] Cam 0 failed. Trying 1..."); cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[WARN] Cam 1 failed. Trying 2..."); cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Could not open any camera.")

    buffer = deque(maxlen=SEQ_LEN)
    ema = None
    counts = defaultdict(int)
    last_count_frame = {c: -10_000 for c in CLASSES}
    frame_idx = 0
    history = deque(maxlen=2000)
    debug = False

    fps, last_ts = 0.0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok: break

        k = detect_kpts_bgr(frame, kprcnn)
        label_txt = "…"; probs_draw = None

        if k is not None:
            kp = normalize_coco17(k)        # (17,3)
            buffer.append(kp)

            if len(buffer) == SEQ_LEN:
                seq = np.asarray(buffer, dtype=np.float32)       # (T,17,3)
                seq_aug = add_velocity(seq).reshape(1, SEQ_LEN, 17*5)  # (1,T,85)

                # standardize if stats are available
                if (mu is not None) and (sigma is not None):
                    seq_aug = (seq_aug - mu) / (sigma + 1e-6)

                raw = clf.predict(seq_aug, verbose=0)[0]         # (3,)
                ema = raw if ema is None else (1.0 - SMOOTH) * ema + SMOOTH * raw
                probs = ema
                probs_draw = probs

                cls_id = int(np.argmax(probs))
                cls_name = CLASSES[cls_id]
                p = float(probs[cls_id])
                label_txt = f"{cls_name} ({p*100:.1f}%)"

                # motion + threshold + cooldown to count
                speed = avg_wrist_speed(seq)
                if debug:
                    print(f"class={cls_name} p={p:.3f} speed={speed:.4f}")

                if p >= THRESHES.get(cls_name, 0.6) and \
                   speed >= MIN_WRIST_SPEED and \
                   (frame_idx - last_count_frame[cls_name] >= COOLDOWN):
                    counts[cls_name] += 1
                    last_count_frame[cls_name] = frame_idx
                    history.append((time.time(), cls_name, round(p,3)))
        else:
            if ema is not None:
                ema = 0.95 * ema
                probs_draw = ema

        # FPS
        now = time.time(); dt = now - last_ts
        if dt > 0: fps = 0.9 * fps + 0.1 * (1.0 / dt)
        last_ts = now

        # UI
        cv2.putText(frame, f"Prediction: {label_txt}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, TEXT, 3, cv2.LINE_AA)

        y0 = 80
        for i, c in enumerate(CLASSES):
            cv2.putText(frame, f"{c}: {counts[c]}", (20, y0 + 32*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT, 2, cv2.LINE_AA)

        draw_prob_bars(frame, probs_draw, x=20, y=y0 + 32*len(CLASSES) + 10)

        cv2.putText(frame, "q:quit  r:reset  s:save  d:debug  "
                           f"FPS:{fps:.1f}", (20, frame.shape[0]-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ACCENT, 2, cv2.LINE_AA)

        cv2.imshow("Keypoint R-CNN + LSTM (no YOLO/MediaPipe)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'):
            counts = defaultdict(int)
            last_count_frame = {c: -10_000 for c in CLASSES}
            ema = None; history.clear()
            print("[INFO] Counters reset.")
        elif key == ord('s'):
            os.makedirs("sessions", exist_ok=True)
            fname = time.strftime("sessions/session_%Y%m%d_%H%M%S.csv")
            with open(fname, "w", newline="") as f:
                w = csv.writer(f); w.writerow(["timestamp","class","prob"])
                w.writerows(history)
            print(f"[INFO] Saved session CSV -> {fname}")
        elif key == ord('d'):
            debug = not debug
            print(f"[INFO] Debug {'ON' if debug else 'OFF'}")

        frame_idx += 1

    if history:
        os.makedirs("sessions", exist_ok=True)
        fname = time.strftime("sessions/session_%Y%m%d_%H%M%S.csv")
        with open(fname, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["timestamp","class","prob"])
            w.writerows(history)
        print(f"[INFO] Auto-saved session CSV -> {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
