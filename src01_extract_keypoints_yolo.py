import os, glob, numpy as np, cv2
from ultralytics import YOLO

DATA_DIR = "data"
OUT_DIR = "extracts"
SEQ_LEN = 30                      # frames per sample (≈1s at 30fps)
MODEL_WEIGHTS = "yolov8n-pose.pt" # try yolov8m-pose.pt for higher accuracy
CLASSES = ["jab", "hook", "uppercut"]

os.makedirs(OUT_DIR, exist_ok=True)
model = YOLO(MODEL_WEIGHTS)

# COCO pose has 17 keypoints; shoulders indices: left=5, right=6
L_SHOULDER, R_SHOULDER = 5, 6

def normalize_keypoints(kpts):
    """
    kpts: (17,3) as [x,y,conf]
    - center by shoulder midpoint
    - scale by shoulder distance
    returns (17,3): [x_norm, y_norm, conf]
    """
    left, right = kpts[L_SHOULDER, :2], kpts[R_SHOULDER, :2]
    center = (left + right) / 2.0
    scale = float(np.linalg.norm(left - right) + 1e-6)
    xy = (kpts[:, :2] - center) / scale
    conf = kpts[:, 2:3]
    return np.concatenate([xy, conf], axis=1)

def video_to_sequences(path, seq_len=SEQ_LEN, step=None):
    if step is None:
        step = seq_len // 2  # 50% overlap
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(source=frame, verbose=False)[0]

        if len(res.keypoints) == 0:
            continue

        # choose the largest person if multiple detected
        if len(res.boxes) > 0:
            areas = (res.boxes.xyxy[:,2]-res.boxes.xyxy[:,0])*(res.boxes.xyxy[:,3]-res.boxes.xyxy[:,1])
            idx = int(np.argmax(areas))
        else:
            idx = 0

        k = res.keypoints.data[idx].cpu().numpy()  # (17,3)
        kp = normalize_keypoints(k)                # (17,3)
        frames.append(kp)

    cap.release()

    frames = np.asarray(frames, dtype=np.float32)  # (T,17,3)
    if len(frames) < seq_len:
        return []

    seqs = []
    for start in range(0, len(frames) - seq_len + 1, step):
        seqs.append(frames[start:start+seq_len])   # (seq_len,17,3)
    return seqs

def main():
    X, y = [], []
    for label, cls in enumerate(CLASSES):
        videos = sorted(glob.glob(os.path.join(DATA_DIR, cls, "*.mp4")))
        print(f"[{cls}] {len(videos)} videos")
        for v in videos:
            seqs = video_to_sequences(v)
            for s in seqs:
                X.append(s)
                y.append(label)

    X = np.asarray(X, dtype=np.float32)  # (N,seq_len,17,3)
    y = np.asarray(y, dtype=np.int64)
    np.save(os.path.join(OUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUT_DIR, "y.npy"), y)
    print("Saved:", os.path.join(OUT_DIR, "X.npy"), X.shape)
    print("Saved:", os.path.join(OUT_DIR, "y.npy"), y.shape)

if __name__ == "__main__":
    main()
