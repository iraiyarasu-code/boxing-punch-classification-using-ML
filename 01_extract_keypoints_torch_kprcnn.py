# 01_extract_keypoints_torch_kprcnn.py
import os, glob, numpy as np, cv2, torch
from torchvision import transforms
from torchvision.models.detection import keypointrcnn_resnet50_fpn

# ---- config ----
DATA_DIR = "data"           # expects data/jab, data/hook, data/uppercut
OUT_DIR  = "extracts"
SEQ_LEN  = 30
CLASSES  = ["jab", "hook", "uppercut"]

# local weights (no internet needed)
LOCAL_WEIGHTS = r"models\weights\keypointrcnn_resnet50_fpn_coco.pth"

device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
to_tensor = transforms.ToTensor()
L_SHOULDER, R_SHOULDER = 5, 6  # COCO-17 indices

def load_kprcnn_local():
    if not os.path.exists(LOCAL_WEIGHTS):
        raise FileNotFoundError(
            f"Missing weights: {LOCAL_WEIGHTS}\n"
            "Download from:\n"
            "  https://download.pytorch.org/models/keypointrcnn_resnet50_fpn_coco-fc266e95.pth\n"
            "and save as keypointrcnn_resnet50_fpn_coco.pth in models\\weights\\"
        )
    print(f"[INFO] Loading Keypoint R-CNN from local file: {LOCAL_WEIGHTS}")
    m = keypointrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    state = torch.load(LOCAL_WEIGHTS, map_location=device)
    m.load_state_dict(state)
    m.to(device).eval()
    return m

@torch.inference_mode()
def detect_keypoints_bgr(frame_bgr, model):
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inp = to_tensor(img_rgb).to(device)  # (3,H,W) float32 [0..1]
    out = model([inp])[0]
    if len(out["keypoints"]) == 0:
        return None
    scores = out["scores"].detach().cpu().numpy()
    idx = int(scores.argmax())
    kpts = out["keypoints"][idx].detach().cpu().numpy()  # (17,3) [x,y,vis]
    # make a [x,y,conf] where conf≈score of the detection
    conf = np.clip(scores[idx], 0.0, 1.0)
    kpts = np.concatenate([kpts[:, :2], np.full((17,1), conf, dtype=np.float32)], axis=1)
    return kpts  # (17,3)

def normalize_keypoints(kpts):
    left, right = kpts[L_SHOULDER,:2], kpts[R_SHOULDER,:2]
    center = (left + right) / 2.0
    scale  = float(np.linalg.norm(left - right) + 1e-6)
    xy   = (kpts[:, :2] - center) / scale
    conf = kpts[:, 2:3]
    return np.concatenate([xy, conf], axis=1)  # (17,3)

def video_to_sequences(path, model, seq_len=SEQ_LEN, step=None):
    if step is None:
        step = seq_len // 2  # 50% overlap
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok: break
        k = detect_keypoints_bgr(frame, model)
        if k is None: continue
        frames.append(normalize_keypoints(k))
    cap.release()
    frames = np.asarray(frames, dtype=np.float32)  # (T,17,3)
    if len(frames) < seq_len:
        return []
    seqs = []
    for start in range(0, len(frames)-seq_len+1, step):
        seqs.append(frames[start:start+seq_len])
    return seqs

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    model = load_kprcnn_local()

    X, y = [], []
    for label, cls in enumerate(CLASSES):
        vids = sorted(glob.glob(os.path.join(DATA_DIR, cls, "*.mp4")))
        print(f"[{cls}] {len(vids)} videos")
        for v in vids:
            for s in video_to_sequences(v, model):
                X.append(s); y.append(label)

    X = np.asarray(X, dtype=np.float32)  # (N,30,17,3)
    y = np.asarray(y, dtype=np.int64)
    np.save(os.path.join(OUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUT_DIR, "y.npy"), y)
    print("Saved:", os.path.join(OUT_DIR, "X.npy"), X.shape)
    print("Saved:", os.path.join(OUT_DIR, "y.npy"), y.shape)

if __name__ == "__main__":
    main()
