
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
import shutil

# =====================================
# 1. CONFIGURATION
# =====================================

SOURCE = Path(
    r"C:\Users\ThinkBook\Downloads\Medical Imagining (CT scan, MRI, X-ray, and Microscopic Imagery) Data\Medical Imagining\Brain Tumor Classification"
)

DESTINATION = Path("dataset")

CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp"
}

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

# =====================================
# 2. CHECK SOURCE
# =====================================

training_source = SOURCE / "Training"
testing_source = SOURCE / "Testing"

if not training_source.exists():
    raise FileNotFoundError(
        f"Training folder not found: {training_source}"
    )

if not testing_source.exists():
    raise FileNotFoundError(
        f"Testing folder not found: {testing_source}"
    )

if DESTINATION.exists():
    raise FileExistsError(
        "The dataset destination already exists. "
        "Check it before running again. "
        "Do not overwrite existing data."
    )

# =====================================
# 3. CREATE DESTINATION
# =====================================

for split in ["train", "val", "test"]:
    for class_name in CLASSES:
        (
            DESTINATION / split / class_name
        ).mkdir(parents=True, exist_ok=True)

# =====================================
# 4. FIND VALID IMAGES
# =====================================

def get_valid_images(folder):

    if not folder.exists():
        raise FileNotFoundError(
            f"Missing class folder: {folder}"
        )

    files = []

    for file in folder.iterdir():

        if not file.is_file():
            continue

        if file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        try:
            with Image.open(file) as image:
                image.verify()

            files.append(file)

        except Exception:
            print("Skipping invalid image:", file.name)

    return files

# =====================================
# 5. SPLIT TRAINING DATA
# =====================================

print("\nPREPARING TRAINING AND VALIDATION DATA")

for class_name in CLASSES:

    source_folder = training_source / class_name

    images = get_valid_images(source_folder)

    if len(images) < 2:
        raise ValueError(
            f"Not enough valid images in {class_name}"
        )

    train_images, val_images = train_test_split(
        images,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE
    )

    # Copy training images

    for image in train_images:

        shutil.copy2(
            image,
            DESTINATION / "train" / class_name / image.name
        )

    # Copy validation images

    for image in val_images:

        shutil.copy2(
            image,
            DESTINATION / "val" / class_name / image.name
        )

    print(
        f"{class_name}: "
        f"Train={len(train_images)}, "
        f"Validation={len(val_images)}"
    )

# =====================================
# 6. COPY ORIGINAL TEST DATA
# =====================================

print("\nCOPYING ORIGINAL TEST DATA")

for class_name in CLASSES:

    source_folder = testing_source / class_name

    images = get_valid_images(source_folder)

    if not images:
        raise ValueError(
            f"No valid test images in {class_name}"
        )

    for image in images:

        shutil.copy2(
            image,
            DESTINATION / "test" / class_name / image.name
        )

    print(
        f"{class_name}: Test={len(images)}"
    )

# =====================================
# 7. FINAL SUMMARY
# =====================================

print("\nDATASET PREPARATION COMPLETED")

for split in ["train", "val", "test"]:

    total = 0

    print(f"\n{split.upper()}")

    for class_name in CLASSES:

        folder = DESTINATION / split / class_name

        count = sum(
            1 for file in folder.iterdir()
            if file.is_file()
        )

        total += count

        print(f"{class_name}: {count}")

    print("TOTAL:", total)

print("\nDataset saved at:", DESTINATION.resolve())
