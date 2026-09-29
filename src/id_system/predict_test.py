"""Run the trained model's predictions on the held-out test split and report per-image results."""

import pandas as pd
import torch
from torch.utils.data import DataLoader

from id_system import (
    BATCH_SIZE,
    IMAGE_DIR,
    METADATA_PATH,
    PlantDataset,
    build_model,
    val_transform,
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    metadata = pd.read_csv(METADATA_PATH)
    num_classes = metadata.shape[0]
    label_to_species = metadata.set_index("label_index")["species"].to_dict()

    model, trained_epochs = build_model(num_classes, device)
    if trained_epochs == 0:
        raise RuntimeError("No trained checkpoint found in processed/model.pt; train the model first.")
    model.eval()

    test_data = PlantDataset(METADATA_PATH, IMAGE_DIR, "test", transform=val_transform)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    correct = 0
    sample_idx = 0
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            predicted_labels = model(X).argmax(1)
            correct += (predicted_labels == y).sum().item()
            for predicted_label in predicted_labels.tolist():
                path, true_label = test_data.samples[sample_idx]
                mark = "OK" if predicted_label == true_label else "WRONG"
                print(
                    f"[{mark}] {path.name}: "
                    f"predicted={label_to_species[predicted_label]!r}, "
                    f"actual={label_to_species[true_label]!r}"
                )
                sample_idx += 1

    accuracy = correct / len(test_data)
    print(f"\nTest Accuracy: {(100 * accuracy):>0.1f}% ({correct}/{len(test_data)})")


if __name__ == "__main__":
    main()
