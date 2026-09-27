import torch
from torch.utils.data import DataLoader
import pandas as pd
from id_system import (
METADATA_PATH,
IMAGE_DIR,
PlantDataset,
build_model,
val_transform
)
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    metadata = pd.read_csv(METADATA_PATH)
    num_classes = metadata.shape[0]
    model, trained_epochs = build_model(num_classes,device)
    dataset = PlantDataset(METADATA_PATH,IMAGE_DIR,"test",val_transform)


    sample_index = 23  # or make this a CLI arg
    image, true_label = dataset[sample_index]
    path, _ = dataset.samples[sample_index]

    image = image.unsqueeze(0).to(device)  # [3, 224, 224] -> [1, 3, 224, 224]

    with torch.no_grad():
        predicted_label = model(image).argmax(1).item()

    print(f"{path.name}: predicted={predicted_label!r}, actual={true_label!r}")




if __name__ == "__main__":
    main()
