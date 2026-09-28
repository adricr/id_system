from pathlib import Path
import pandas as pd
import csv
CURRENT_FILE_PATH = Path(__file__)
PROJECT_DIR = CURRENT_FILE_PATH.parents[2]
DATASET_FOLDER = PROJECT_DIR / "plants1"
METADATA = DATASET_FOLDER / "metadata.csv"

SPLITS = ("train", "test", "valid")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

HEADER = ['label_index','dir_id','species','common_name','train_count','valid_count','test_count','total_count']
def update_metadata_func():
    DIRTOSPECIES = [
        {"dir_id": "2765221", "species": "Tradescantia pallida", "common_name": "Purple heart"},
        {"dir_id": "2868241", "species": "Monstera Deliciosa", "common_name": "Swiss cheese plant"},
        {"dir_id":"2868323", "species":"Epipremnum aureum", "common_name":"Golden pothos"},
        {"dir_id":"2869680", "species":"Spathiphyllum wallisii", "common_name":"Peace lily"},
        {"dir_id":"2927096", "species":"Ocimum basilicum", "common_name":"Basil"},
        {"dir_id":"2930137", "species":"Solanum lycopersicum L.", "common_name":"Tomato"},
        {"dir_id":"2985944", "species":"Kalanchoe Blossfeldiana", "common_name":"Flaming Katy"},
        {"dir_id":"5361899", "species":"Ficus Lyrata", "common_name":"Fiddle-leaf fig"},
        {"dir_id":"9388529", "species":"Haworthiopsis attenuata", "common_name":"Zebra haworthia"},
        {"dir_id":"5362063", "species":"Crassula Ovata", "common_name":"Jade plant"},
        {"dir_id":"8315197", "species":"Echeveria Elegans", "common_name":"Mexican snowball"},
    ]
    data = []
    with open(METADATA,'r') as reading:
        for line in csv.DictReader(reading):
            data.append(line)
    for split in SPLITS:
        current_split = DATASET_FOLDER/split
        for label in current_split.iterdir():
            if not label.is_dir():
                continue
            file_counter = 0
            for file in label.iterdir():
                if file.suffix.lower() in IMAGE_EXTENSIONS:
                    file_counter += 1
            for row in data:
                if row.get("dir_id") == label.name:
                    row.update({f"{split}_count":file_counter})
                    break
            else:
                print(f"Warning: {split}/{label.name} is not in {METADATA.name}, skipped")

    for names in DIRTOSPECIES:
        for row in data:
            if names.get("dir_id") == row.get("dir_id"):
                row.update({"species":names.get("species"), "common_name": names.get("common_name")})

    for row in data:
        row.update({"total_count": int(row.get("train_count"))+int(row.get("valid_count"))+int(row.get("test_count"))})

    with open(METADATA,'w',newline='') as writing:
        writer = csv.DictWriter(writing, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(data)



