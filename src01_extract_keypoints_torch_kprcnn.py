# src/01_extract_keypoints_torch_kprcnn.py
import os, glob, numpy as np, cv2, torch
from torchvision import transforms
from torchvision.models.detection import keypointrcnn_resnet50_fpn

DATA_DIR = "data"
OUT_DIR = "extracts"
SEQ_LEN = 30
CLASSES = ["jab", "hook", "uppercut"]

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
to_tensor = transforms.ToTensor()

# ---- load pretrained COCO Keypoint R-CNN (will download once) ----
@torch.inference_mode()
def load_kprcnn():
    print("[INFO] Loading TorchVision Keypoint R-CNN with DEFAULT weights (will download if not cached)…")
    m = keypointrcnn_resnet50_fpn(weights="DEFAULT")
    return m.to(device).eval()

model = load_kprcnn()
