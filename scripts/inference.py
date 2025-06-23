import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import pandas as pd

# --- DATASET ---
class LateFusionTestDataset(Dataset):
    def __init__(self, features_rgb_dir, features_radar_dir, classes):
        self.features_rgb_dir = Path(features_rgb_dir)  # senza /test aggiunto
        self.features_radar_dir = Path(features_radar_dir)
        self.classes = classes
        self.samples = []

        # Radar files direttamente nella cartella
        radar_files = {f.stem: f for f in self.features_radar_dir.glob("*.npy")}

        for rgb_file in self.features_rgb_dir.glob("*.npy"):
            stem = rgb_file.stem
            if stem in radar_files:
                radar_file = radar_files[stem]
                sample_id = stem  # qui prendi il nome completo senza estensione
                self.samples.append((sample_id, rgb_file, radar_file))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id, rgb_path, radar_path = self.samples[idx]
        rgb_feat = np.load(rgb_path)
        radar_feat = np.load(radar_path)
        features = np.concatenate([rgb_feat, radar_feat], axis=1)
        features = torch.tensor(features, dtype=torch.float32)
        return sample_id, features

# --- MODELLO ---
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=2144, hidden_dim=512, num_layers=2, num_classes=126, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        return self.classifier(hn[-1])

# --- INFERENZA ---
def run_inference(features_rgb_dir, features_radar_dir, model_path, batch_size=64):
    with open("class_ids.txt", "r") as f:
        classes = [line.strip() for line in f if line.strip()]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔍 Using device: {device}")

    test_dataset = LateFusionTestDataset(features_rgb_dir, features_radar_dir, classes)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMClassifier(num_classes=len(classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    sample_ids = []
    predicted_classes = []

    with torch.no_grad():
        for batch in test_loader:
            ids, features = batch
            features = features.to(device)
            outputs = model(features)
            _, preds = torch.max(outputs, dim=1)
            sample_ids.extend(ids)
            predicted_classes.extend([classes[i] for i in preds.cpu().numpy()])

    df = pd.DataFrame({
        "Id": sample_ids,
        "PredictedClass": predicted_classes
    })
    df.to_csv("dopo_22_final_test_predictions.csv", index=False)
    print("✅ Predizioni salvate in final_final_test_predictions.csv")

# --- MAIN ---
if __name__ == "__main__":
    features_rgb_dir = "/content/content/features_RGB_test/features_RGB_test"
    features_radar_dir = "/content/features_radar_test"
    model_path = "/content/best_model.pth"
    run_inference(features_rgb_dir, features_radar_dir, model_path)
