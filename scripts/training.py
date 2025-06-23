import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

class LateFusionDataset(Dataset):
    def __init__(self, features_rgb_dir, features_radar_dir, classes, split="train"):
        """
        features_rgb_dir: Path cartella con features RGB (.npy)
        features_radar_dir: Path cartella con features radar (.npy)
        classes: lista nomi classi (cartelle)
        split: "train" o "val"
        """
        self.features_rgb_dir = Path(features_rgb_dir) / split
        self.features_radar_dir = Path(features_radar_dir) / split
        self.classes = classes

        self.samples = []  # lista di (rgb_path, radar_path, label)
        for label, class_name in enumerate(classes):
            rgb_class_dir = self.features_rgb_dir / class_name
            radar_class_dir = self.features_radar_dir / class_name

            # Controlla file comuni in entrambe le cartelle (allineamento)
            rgb_files = {f.stem for f in rgb_class_dir.glob("*.npy")}
            radar_files = {f.stem for f in radar_class_dir.glob("*.npy")}
            common_files = rgb_files.intersection(radar_files)

            for sample_name in sorted(common_files):
                rgb_path = rgb_class_dir / f"{sample_name}.npy"
                radar_path = radar_class_dir / f"{sample_name}.npy"
                self.samples.append((rgb_path, radar_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rgb_path, radar_path, label = self.samples[idx]
        rgb_feat = np.load(rgb_path)        # (30, 2048)
        radar_feat = np.load(radar_path)    # (30, 96)

        # Concatenate feature per frame
        features = np.concatenate([rgb_feat, radar_feat], axis=1) 

        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return features, label



import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=2144, hidden_dim=512, num_layers=2, num_classes=10, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, seq_len=30, feature_dim=2304)
        _, (hn, _) = self.lstm(x)  # hn: (num_layers, batch, hidden_dim)
        out = hn[-1]               # usa ultimo layer hidden (batch, hidden_dim)
        out = self.classifier(out) # (batch, num_classes)
        return out



import torch
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt

# --- Funzioni di training e validazione ---
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * features.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# --- Setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"

# Cartelle dataset (modifica se serve)
rgb_root = Path("/content/content/features_RGB/content/features_RGB")
radar_root = Path("/content/content/features_radar/content/features_radar")

# Estrazione classi da train RGB (assumendo che in train ci siano cartelle per ogni classe)
classes = sorted([d.name for d in (rgb_root / "train").iterdir() if d.is_dir()])

# Assumo che LateFusionDataset e LSTMClassifier siano definiti da te e funzionino correttamente
train_dataset = LateFusionDataset(str(rgb_root), str(radar_root), classes, split="train")
val_dataset = LateFusionDataset(str(rgb_root), str(radar_root), classes, split="val")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model = LSTMClassifier(num_classes=len(classes)).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# --- Training Loop ---
num_epochs = 150
best_val_acc = 0

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch:02d} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
