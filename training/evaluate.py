import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = "dataset"
MODEL_PATH = "models/best_model.pth"
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Uses 'test' folder if present, otherwise falls back to 'val'
split = "test" if os.path.exists(os.path.join(DATA_DIR, "test")) else "val"
dataset = datasets.ImageFolder(os.path.join(DATA_DIR, split), transform=transform)
loader = DataLoader(dataset, batch_size=16, shuffle=False)
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
CLASSES = checkpoint.get("class_names", ["glioma", "meningioma", "notumor", "pituitary"])
IMG_SIZE = checkpoint.get("image_size", 224)

model = models.efficientnet_b0()
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device).eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in loader:
        outputs = model(images.to(device))
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

print(f"--- Evaluation on '{split}' set ---")
print(classification_report(all_labels, all_preds, target_names=CLASSES))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Brain MRI Confusion Matrix")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
print("Saved confusion matrix to models/confusion_matrix.png")