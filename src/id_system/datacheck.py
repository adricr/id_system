from id_system import PlantDataset, METADATA_PATH, IMAGE_DIR, train_transform
from torch.utils.data import DataLoader, Dataset
def main() -> None:
    check_data = PlantDataset(METADATA_PATH,IMAGE_DIR, "train", train_transform)
    # for sample in range(0,5):
    #     print(check_data.__getitem__(sample))
    check_dataloader = DataLoader(check_data,10,True)
    for batch, (X, y) in enumerate(check_dataloader):
        print(f"batch:{batch}")
        print(f"X:{X}")
        print(f"y:{y}")
if __name__ == '__main__':
    main()