import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import ImageReadMode, decode_image
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision.transforms import v2
from early_stopping import EarlyStopping

# Repo layout: <project_root>/src/id_system/id_system.py, dataset at <project_root>/plants1
PROJECT_DIR = Path(__file__).resolve().parents[2]
IMAGE_DIR = PROJECT_DIR / "plants1"
PROCESSED_DIR = PROJECT_DIR / "processed"
METADATA_PATH = IMAGE_DIR / "metadata.csv"
model_path = PROCESSED_DIR / "model_large.pt"
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 200


class PlantDataset(Dataset):
    """Loads plant images from IMAGE_DIR/<split>/<dir_id>/ as listed in metadata.csv."""

    def __init__(self, metadata_path, image_dir, split, transform=None, target_transform=None):
        self.image_dir = Path(image_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        metadata = pd.read_csv(metadata_path)
        self.samples = []
        for _, row in metadata.iterrows():
            label = int(row["label_index"])
            class_dir = self.image_dir / split / str(row["dir_id"])
            for filename in sorted(os.listdir(class_dir)):
                self.samples.append((class_dir / filename, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = decode_image(str(path), mode=ImageReadMode.RGB)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


train_transform = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_transform = v2.Compose([
    v2.Resize(size=(224, 224), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_model(num_classes, device):
    if model_path.exists():
        model = mobilenet_v3_large(weights=None)
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        trained_epochs = checkpoint.get("epoch", 0)
        print(f"Loaded model from {model_path} (trained for {trained_epochs} epochs so far)")
    else:
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1, progress=True)
        for p in model.parameters():
            p.requires_grad = False
        model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)
        trained_epochs = 0
    return model.to(device), trained_epochs


def train_loop(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    correct = 0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss_value, current = loss.item(), batch * BATCH_SIZE + len(X)

            print(f"train loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")


def val_loop(dataloader, model, loss_fn, device):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    val_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")
    return val_loss

def main() -> None:
    print("PyTorch:", torch.__version__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(METADATA_PATH)
    num_classes = metadata.shape[0]

    training_data = PlantDataset(METADATA_PATH, IMAGE_DIR, "train", transform=train_transform)
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    val_data = PlantDataset(METADATA_PATH, IMAGE_DIR, "valid", transform=val_transform)
    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)

    model, trained_epochs = build_model(num_classes, device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    early_stopping = EarlyStopping(4,0.01)
    total_epochs = 0
    print(f"Let's run {EPOCHS} epochs")
    for epoch in range(trained_epochs + 1, trained_epochs + EPOCHS + 1):
        print(f"Epoch {epoch}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, device)
        early_stopping(val_loop(val_dataloader, model, loss_fn, device), model)
        if early_stopping.early_stop:
            total_epochs = epoch
            print("Early stop to avoid overfitting")
            break
        total_epochs = trained_epochs + EPOCHS
    early_stopping.load_best_model(model)
    print(f"Done! Total epochs trained: {total_epochs}")
    torch.save({
        "epoch": total_epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, model_path)
    print("Saved model to", model_path)


if __name__ == "__main__":
    main()
