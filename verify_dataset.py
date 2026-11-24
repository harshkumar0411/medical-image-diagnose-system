
from pathlib import Path

DATASET = Path("dataset")

CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

for split in ["train", "val", "test"]:

    print(f"\n{split.upper()}")

    for class_name in CLASSES:

        folder = DATASET / split / class_name

        if not folder.exists():
            print("ERROR: Missing", folder)
            continue

        count = len([
            f for f in folder.iterdir()
            if f.is_file()
        ])

        print(f"{class_name}: {count} images")

print("\nDataset verification completed.")
