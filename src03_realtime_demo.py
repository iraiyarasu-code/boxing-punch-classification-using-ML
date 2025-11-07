# src/03_realtime_demo.py
import os
import time
import csv
from collections import deque, defaultdict

import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

# -------------------- Config --------------------
CLASSES = ["jab", "hook", "uppercut"]
YOLO_WEIGHTS = "yolov8n-pose.pt"   # try 'yolov8m-pose.pt' for higher accuracy
YOLO_IMGSZ = 320                   # lower = faster, 320/416/512
CAM_INDEX = 0                      # change if you have multiple cameras
SEQ_LEN = 30                       # frames per sequence to LSTM
THRESH = 0.75                      # min prob to count a punch
ALPHA = 0.6                        # EMA smoothing factor (higher = smoother)
COOLDOWN = 8                       # frames before counting same class again
MODEL_PATHS = [
    "models/punch_lstm_best.h5",
    "models/punch_lstm.h5",
    "models/punch_lstm.keras",
]  # will load the first that exists

TEXT_COLOR = (0, 255, 0)
WARN_COLOR = (0, 165, 255)
# ------------------------------------------------

L_SHOULDER, R_SHOULDER = 5, 6  # COCO indices


def normalize_keypoints(kpts):
    """
    kpts: (17,3) [x,y,conf] from YOLO-Pose
    center by shoulder midpoint, scale by shoulder distance
    returns (17,3)
    """
    left, right = kpts[L_SHOULDER, :2], kpts[R_SHOULDER, :2]
    center = (left + right) / 2.0
    scale = float(np.linalg.norm(left - right) + 1e-6)
    xy = (kpts[:, :2] - center) / scale
    conf = kpts[:, 2:3]
    return np.concatenate([xy, conf], axis=1)


def add_velocity(seq_173):
    """
    seq_173: (T,17,3) -> concat velocity (dx,dy) to get (T,17,5)
    """
    vel = np.diff(seq_173[..., :2], axis=0, prepend=seq_173[:1, ..., :2])
    conf = seq_173[..., 2:3]
    xy = seq_173[..., :2]
    return np.concatenate([xy, conf, vel], axis=-1)  # (T,17,5)


def load_classifier():
    for p in MODEL_PATHS:
        if os.path.exists(p):
            print(f"[INFO] Loading classifier: {p}")
            model = tf.keras.models.load_model(p)
            # Not strictly required for predict(), but silences TF warning:
            model.compile(optimizer="adam",
                          loss="sparse_categorical_crossentropy",
                          metrics=["accuracy"])
            return model
    raise FileNotFoundError(
        f"No model found. Expected one of: {MODEL_PATHS}"
    )


def main():
    # YOLO pose model
    pose_model = YOLO(YOLO_WEIGHTS)
    pose_model.overrides["imgsz"] = YOLO_IMGSZ

    # LSTM classifier
    clf = load_classifier()

    # camera
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Try CAM_INDEX=1 or 2.")

    buffer = deque(maxlen=SEQ_LEN)          # last T frames of (17,3)
    ema = None                               # EMA probs
    counts = defaultdict(int)                # live counters per class
    last_count_frame = {c: -999999 for c in CLASSES}
    frame_idx = 0
    history = deque(maxlen=2000)             # session events (ts, class, prob)

    last_fps_ts = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # --- YOLO pose inference
        res = pose_model.predict(source=frame, verbose=False)[0]

        label_txt = "…"
        # choose largest person if present
        if len(res.keypoints) > 0:
            if len(res.boxes) > 0:
                areas = (res.boxes.xyxy[:, 2] - res.boxes.xyxy[:, 0]) * \
                        (res.boxes.xyxy[:, 3] - res.boxes.xyxy[:, 1])
                idx = int(np.argmax(areas))
            else:
                idx = 0

            k = res.keypoints.data[idx].cpu().numpy()  # (17,3)
            kp = normalize_keypoints(k)                # (17,3)
            buffer.append(kp)

            # if we have T frames -> classify
            if len(buffer) == SEQ_LEN:
                seq = np.asarray(buffer, dtype=np.float32)     # (T,17,3)
                seq = add_velocity(seq).reshape(1, SEQ_LEN, 17 * 5)  # (1,T,85)

                raw = clf.predict(seq, verbose=0)[0]  # (num_classes,)
                ema = raw if ema is None else ALPHA * ema + (1.0 - ALPHA) * raw
                probs = ema

                cls_id = int(np.argmax(probs))
                cls_name = CLASSES[cls_id]
                p = float(probs[cls_id])
                label_txt = f"{cls_name} ({p*100:.1f}%)"

                # counting with cooldown + threshold
                if p >= THRESH and (frame_idx - last_count_frame[cls_name] >= COOLDOWN):
                    counts[cls_name] += 1
                    last_count_frame[cls_name] = frame_idx
                    history.append((time.time(), cls_name, round(p, 3)))

        else:
            # if no person detected, slowly decay EMA so prediction fades
            if ema is not None:
                ema = 0.95 * ema

        # --- FPS calc
        now = time.time()
        dt = now - last_fps_ts
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)
        last_fps_ts = now

        # --- UI draw
        cv2.putText(frame, f"Prediction: {label_txt}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, TEXT_COLOR, 3, cv2.LINE_AA)

        y0 = 80
        for i, c in enumerate(CLASSES):
            cv2.putText(frame, f"{c}: {counts[c]}", (20, y0 + 32 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2, cv2.LINE_AA)

        cv2.putText(frame, f"FPS: {fps:.1f}  |  q:quit  r:reset  s:save CSV",
                    (20, y0 + 32 * len(CLASSES) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, WARN_COLOR, 2, cv2.LINE_AA)

        cv2.imshow("YOLO-Pose + LSTM (with counters)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            counts = defaultdict(int)
            last_count_frame = {c: -999999 for c in CLASSES}
            ema = None
            history.clear()
            print("[INFO] Counters reset.")
        elif key == ord('s'):
            # save quick session CSV
            os.makedirs("sessions", exist_ok=True)
            fname = time.strftime("sessions/session_%Y%m%d_%H%M%S.csv")
            with open(fname, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestamp", "class", "prob"])
                w.writerows(history)
            print(f"[INFO] Saved session CSV -> {fname}")

        frame_idx += 1

    # on exit, also auto-save a summary CSV
    if history:
        os.makedirs("sessions", exist_ok=True)
        fname = time.strftime("sessions/session_%Y%m%d_%H%M%S.csv")
        with open(fname, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "class", "prob"])
            w.writerows(history)
        print(f"[INFO] Auto-saved session CSV -> {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
