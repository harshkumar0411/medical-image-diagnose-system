
from pathlib import Path
from collections import Counter

# CHANGE THIS PATH TO YOUR EXTRACTED DATASET LOCATION
DATASET_PATH = Path(
    r"C:\Users\ThinkBook\Downloads\Medical Imagining (CT scan, MRI, X-ray, and Microscopic Imagery) Data\Medical Imagining"
)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".tif", ".tiff",
    ".dcm"
}

print("=" * 60)
print("MEDICAL DATASET INSPECTION")
print("=" * 60)

if not DATASET_PATH.exists():
    print("ERROR: Dataset folder does not exist!")
    print("Check the DATASET_PATH variable.")
    raise SystemExit

print("\nDataset location:")
print(DATASET_PATH)

# 1. List top-level folders
print("\n1. MAIN FOLDERS")

for item in sorted(DATASET_PATH.iterdir()):
    if item.is_dir():
        print("[FOLDER]", item.name)
    else:
        print("[FILE]", item.name)

# 2. Count image formats
print("\n2. IMAGE FORMAT COUNTS")

extension_counts = Counter()

for file in DATASET_PATH.rglob("*"):
    if file.is_file():
        if file.suffix.lower() in IMAGE_EXTENSIONS:
            extension_counts[file.suffix.lower()] += 1

for extension, count in extension_counts.most_common():
    print(extension, ":", count)

# 3. Find modality-related folders
print("\n3. POSSIBLE MEDICAL IMAGE FOLDERS")

keywords = [
    "xray", "x-ray", "chest",
    "ct", "mri", "microscopic",
    "microscopy", "normal",
    "pneumonia", "covid", "tuberculosis"
]

for folder in DATASET_PATH.rglob("*"):
    if folder.is_dir():
        if any(word in folder.name.lower()
               for word in keywords):
            print(folder)

# 4. Count images by their immediate parent folder
print("\n4. IMAGE COUNTS BY FOLDER")

folder_counts = Counter()

for file in DATASET_PATH.rglob("*"):
    if file.is_file():
        if file.suffix.lower() in IMAGE_EXTENSIONS:
            folder_counts[file.parent] += 1

for folder, count in folder_counts.most_common(50):
    print(f"{folder}: {count} images")

# 5. Find annotation files
print("\n5. POSSIBLE LABEL / ANNOTATION FILES")

for file in DATASET_PATH.rglob("*"):
    if file.is_file():
        if file.suffix.lower() in {
            ".csv", ".json", ".xml",
            ".txt", ".xlsx"
        }:
            print(file)

print("\n" + "=" * 60)
print("INSPECTION COMPLETED")
print("=" * 60)
