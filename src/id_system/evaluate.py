"""Confusion matrix and per-class recall on the validation split."""

import pandas as pd
import torch
from torch.utils.data import DataLoader

from id_system_large import (
    BATCH_SIZE,
    IMAGE_DIR,
    METADATA_PATH,
    PlantDataset,
    build_model,
    val_transform,
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = pd.read_csv(METADATA_PATH)
    num_classes = metadata.shape[0]
    names = metadata.sort_values("label_index")["species"].tolist()

    model, trained_epochs = build_model(num_classes, device)
    if trained_epochs == 0:
        raise RuntimeError("No trained checkpoint found; train the model first.")
    model.eval()

    data = PlantDataset(METADATA_PATH, IMAGE_DIR, "test", transform=val_transform)
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=False)

    # rows = true label, columns = predicted label
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    with torch.no_grad():
        for X, y in loader:
            preds = model(X.to(device)).argmax(1).cpu()
            for t, p in zip(y.tolist(), preds.tolist()):
                cm[t, p] += 1

    short = [n.split()[0][:10] for n in names]  # keep columns narrow
    print(pd.DataFrame(cm.numpy(), index=short, columns=short).to_string())

    recall = cm.diag() / cm.sum(1).clamp(min=1)
    print("\nPer-class recall:")
    for name, r, n in zip(names, recall.tolist(), cm.sum(1).tolist()):
        print(f"  {name:<28} {100 * r:5.1f}%  (n={n})")

    print(f"\nOverall accuracy: {100 * cm.diag().sum().item() / cm.sum().item():.1f}%")

    off = cm.clone()
    off.fill_diagonal_(0)
    print("\nMost common confusions (true -> predicted):")
    for idx in off.flatten().argsort(descending=True)[:5].tolist():
        t, p = divmod(idx, num_classes)
        if off[t, p] == 0:
            break
        print(f"  {names[t]} -> {names[p]}: {off[t, p].item()}")


if __name__ == "__main__":
    main()
