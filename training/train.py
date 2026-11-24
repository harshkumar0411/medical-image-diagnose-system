
import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ======================================
# CONFIGURATION
# ======================================

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.0001

DATA_DIR = "dataset"
MODEL_DIR = "models"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

os.makedirs(MODEL_DIR, exist_ok=True)

print("Device:", DEVICE)

# ======================================
# IMAGE TRANSFORMS
# ======================================

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ======================================
# LOAD DATASETS
# ======================================

train_data = datasets.ImageFolder(
    DATA_DIR + "/train",
    transform=train_transform
)

val_data = datasets.ImageFolder(
    DATA_DIR + "/val",
    transform=eval_transform
)

test_data = datasets.ImageFolder(
    DATA_DIR + "/test",
    transform=eval_transform
)

class_names = train_data.classes

print("Classes:", class_names)

for dataset in [val_data, test_data]:

    if dataset.classes != class_names:
        raise ValueError("Dataset classes do not match!")

if len(class_names) != 4:
    raise ValueError("Expected exactly four classes.")

if min(len(train_data), len(val_data), len(test_data)) == 0:
    raise ValueError("A dataset partition is empty.")

# ======================================
# DATA LOADERS
# ======================================

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# ======================================
# LOAD PRETRAINED MODEL
# ======================================

print("\nLoading EfficientNet-B0...")

weights = models.EfficientNet_B0_Weights.DEFAULT

model = models.efficientnet_b0(
    weights=weights
)

# Freeze feature extractor initially

#for parameter in model.features.parameters():
    #parameter.requires_grad = False

# Replace final classification layer

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features,
    len(class_names)
)

model = model.to(DEVICE)

# ======================================
# LOSS AND OPTIMIZER
# ======================================

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=3e-4,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)

# ======================================
# TRAINING FUNCTION
# ======================================

def run_epoch(loader, training):

    model.train() if training else model.eval()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):

            outputs = model(images)

            loss = criterion(outputs, labels)

            if training:
                loss.backward()
                optimizer.step()

        predictions = outputs.argmax(dim=1)

        total_loss += loss.item() * labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    return total_loss / total, correct / total

# ======================================
# MAIN TRAINING LOOP
# ======================================

best_accuracy = -1
best_weights = copy.deepcopy(model.state_dict())

print("\nStarting training...")

for epoch in range(EPOCHS):

    train_loss, train_acc = run_epoch(
        train_loader, True
    )

    val_loss, val_acc = run_epoch(
        val_loader, False
    )
    
    scheduler.step(val_loss)

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_acc:.4f}"
    )

    print(
        f"Validation Loss: {val_loss:.4f} | "
        f"Validation Accuracy: {val_acc:.4f}"
    )

    if val_acc > best_accuracy:

        best_accuracy = val_acc

        best_weights = copy.deepcopy(
            model.state_dict()
        )

# ======================================
# SAVE MODEL
# ======================================

model.load_state_dict(best_weights)

checkpoint = {
    "model_state_dict": model.state_dict(),
    "class_names": class_names,
    "image_size": IMAGE_SIZE
}

torch.save(
    checkpoint,
    MODEL_DIR + "/best_model.pth"
)

print("\nTRAINING COMPLETED")

print("Best validation accuracy:", best_accuracy)

print("Model saved to models/best_model.pth")
