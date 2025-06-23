import os
import cv2
import torch
import numpy as np
from pathlib import Path
from torchvision import models, transforms
import torch.nn as nn
from tqdm import tqdm
import zipfile
import shutil

from google.colab import drive

#DRIVE_PATH = Path("/content/drive/MyDrive/")
#drive.mount('/content/drive')
#processed_drive = DRIVE_PATH / "processed_train.txt"
#if processed_drive.exists():
    #shutil.copy(processed_drive, "processed_train.txt")

# === CONFIG ===
SAMPLES_DIR = Path("/content/my_local_train_data_raw")
RGB_OUT = Path("/content/features_RGB_train")
RADAR_OUT = Path("/content/features_radar_train")
ZIP_RGB = Path("/content/features_RGB_train.zip")
ZIP_RADAR = Path("/content/features_radar_train.zip")

PROCESSED_FILE = Path("/content/processed_train.txt")
BATCH_SIZE = 100
NUM_FRAMES = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RGB_OUT.mkdir(parents=True, exist_ok=True)
RADAR_OUT.mkdir(parents=True, exist_ok=True)
DRIVE_PATH.mkdir(parents=True, exist_ok=True)

# === LOAD PROCESSED LIST ===
if not PROCESSED_FILE.exists():
    PROCESSED_FILE.touch()

with open(PROCESSED_FILE, "r") as f:
    already_done = set(line.strip() for line in f if line.strip())

print(f"{len(already_done)} samples already processed.")

# === MODELS ===
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet = nn.Sequential(*list(resnet.children())[:-1])
resnet.to(DEVICE).eval()

class RadarCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

radar_cnn = RadarCNN().to(DEVICE).eval()

# === TRANSFORMS ===
rgb_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# === UTILS ===
def extract_frames(path, n=NUM_FRAMES, gray=False):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = list(np.linspace(0, total-1, n, dtype=int)) if total >= n else list(range(total))
    frames = []
    i = 0
    while len(frames) < len(idxs):
        ret, f = cap.read()
        if not ret:
            break
        if i in idxs:
            if gray:
                f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            else:
                f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            frames.append(f)
        i += 1
    cap.release()
    if len(frames) < n:
        pad = np.zeros_like(frames[0]) if frames else np.zeros((224,224,3), dtype=np.uint8)
        frames += [pad.copy() for _ in range(n - len(frames))]
    return frames[:n]

def process_sample(sd):
    sid = sd.name                          # es. SAMPLE_001
    class_dir = sd.parent.name             # es. 0_a
    out_rgb_class_dir = RGB_OUT / class_dir
    out_radar_class_dir = RADAR_OUT / class_dir
    out_rgb_class_dir.mkdir(parents=True, exist_ok=True)
    out_radar_class_dir.mkdir(parents=True, exist_ok=True)

    out_rgb_file = out_rgb_class_dir / f"{sid}.npy"
    out_radar_file = out_radar_class_dir / f"{sid}.npy"

    # Skip se già processato
    if f"{class_dir}/{sid}" in already_done:
        return

    # RGB
    if not out_rgb_file.exists():
        p_rgb = next(sd.glob("*RGB.mkv"), None)
        if p_rgb:
            fr = extract_frames(p_rgb)
            ten = torch.stack([rgb_tf(f) for f in fr]).to(DEVICE)
            with torch.no_grad():
                feat = resnet(ten).squeeze(-1).squeeze(-1).cpu().numpy()
            np.save(out_rgb_file, feat)

    # Radar
    if not out_radar_file.exists():
        radar_feats = []
        for i in [1, 2, 3]:
            pr = next(sd.glob(f"*RDM{i}.mp4"), None)
            if pr:
                fr = extract_frames(pr, gray=True)
                arr = np.array(fr)
                ten = torch.tensor(arr).unsqueeze(1).float() / 255.0
                ten = nn.functional.interpolate(ten, size=(224, 224)).to(DEVICE)
                with torch.no_grad():
                    f = radar_cnn(ten).cpu().numpy()
                radar_feats.append(f)
        if radar_feats:
            rf = np.concatenate(radar_feats, axis=1)
            np.save(out_radar_file, rf)

    # Salva come classe/SAMPLE_ID nel file processed
    with open(PROCESSED_FILE, "a") as f:
        f.write(f"{class_dir}/{sid}\n")
    already_done.add(f"{class_dir}/{sid}")


def zip_and_upload_incremental():
    # Update RGB zip
    # Update RGB zip
    with zipfile.ZipFile(DRIVE_PATH / ZIP_RGB.name, 'a', compression=zipfile.ZIP_DEFLATED) as zipf:
        for file in RGB_OUT.rglob("*.npy"):
            arcname = file.relative_to(RGB_OUT)
            if str(arcname) not in zipf.namelist():
                zipf.write(file, arcname=str(arcname))


    # Update Radar zip
    with zipfile.ZipFile(DRIVE_PATH / ZIP_RADAR.name, 'a', compression=zipfile.ZIP_DEFLATED) as zipf:
        for file in RADAR_OUT.rglob("*.npy"):
            arcname = file.relative_to(RADAR_OUT)
            if str(arcname) not in zipf.namelist():
                zipf.write(file, arcname=str(arcname))
    shutil.copy("processed_train.txt", DRIVE_PATH / "processed_train.txt")
    # Cancella solo i file appena processati
    for f in RGB_OUT.rglob("*.npy"):
        f.unlink()
    for f in RADAR_OUT.rglob("*.npy"):
        f.unlink()

# === MAIN ===
samples = sorted(SAMPLES_DIR.glob("*/*/"))  # classe/SAMPLE_*/
print(f"{len(already_done)} samples already processed.")

for i in range(0, len(samples), BATCH_SIZE):
    batch = samples[i:i+BATCH_SIZE]
    print(f"Processing batch {i//BATCH_SIZE+1} ({len(batch)} samples)...")
    for sd in tqdm(batch):
        process_sample(sd)
    print("Zipping and uploading to Drive...")
    zip_and_upload_incremental()

print("✅ All done!")
