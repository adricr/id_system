import torch
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from id_system_large import (
METADATA_PATH,
PROJECT_DIR,
PlantDataset,
build_model,
val_transform
)
TESTIMAGEPATH = PROJECT_DIR/"test"
IMAGENAME = "philoden.jpg"
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    metadata = pd.read_csv(METADATA_PATH, index_col="label_index")
    num_classes = metadata.shape[0]
    model, trained_epochs = build_model(num_classes,device)
    image = decode_image(str(TESTIMAGEPATH / IMAGENAME), mode=ImageReadMode.RGB, apply_exif_orientation=True)
    test_transform = v2.Compose([
        v2.Resize(size=256, antialias=True),
        v2.CenterCrop(size=256),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = test_transform(image)
    show_image(image)
    image = image.unsqueeze(0).to(device)  # [3, 224, 224] -> [1, 3, 224, 224]
    model.eval()
    # with torch.no_grad():
    #     predicted_label = model(image).argmax(1).item()

    with torch.no_grad():
        logits = model(image)
    probs = torch.softmax(logits, dim=1).squeeze(
        0).cpu()  # [num_classes]

    top_probs, top_idxs = probs.topk(3)
    print(f"{IMAGENAME}:")
    for p, idx in zip(top_probs.tolist(), top_idxs.tolist()):
        row = metadata.loc[idx]
        print(f"  {p:6.1%}  {row['species']} ({row['common_name']})")

    # print(f"{IMAGENAME}: predicted={predicted_label!r}")
    # print(metadata)


def show_image(tensor, title=None):
    img = tensor.squeeze(
        0).cpu()  # drop the batch dim if present
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (img * std + mean).clamp(0,
                                   1)  # undo Normalize
    plt.imshow(img.permute(1, 2,
                           0))  # [C,H,W] -> [H,W,C]
    plt.axis("off")
    if title:
        plt.title(title)
    plt.show()

if __name__ == "__main__":
    main()
