import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import glob
from tqdm import tqdm
import random
from collections import Counter
from pathlib import Path

# Import your model (keep same API you used before)
from model import GlaucomaModel

# -----------------------
# Repro & config
# -----------------------
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
IMAGE_SIZE = 224
OUTPUT_DIR = "model/output"
MODEL_NAME = "efficientnet_b0"
RAW_ROOT = Path("data") / "raw"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------
# Transforms
# -----------------------
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -----------------------
# Helper: infer label from path
# -----------------------
# Heuristics: check path components for these keywords
LABEL_KEYWORDS = {
    "glaucoma": ["glaucoma", "g", "glaucoma_positive", "glaucoma-pos", "glaucoma_pos"],
    "normal": ["normal", "healthy", "h", "non-glaucoma", "non_glaucoma", "non-glaucomatous"],
    "dr": ["diabetic", "dr", "diabetic_retinopathy", "diabetic-retinopathy"],
    "pathological": ["pathological", "disease", "abnormal", "lesion", "pathology"]
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

def infer_label_from_path(path):
    p = str(path).lower()
    # check exact keywords first
    for label, keys in LABEL_KEYWORDS.items():
        for k in keys:
            if f"/{k}/" in f"/{p}/" or f"_{k}" in p or f"-{k}" in p or f"{k}." in p:
                return label
    # fallback heuristics: check folder names
    for part in Path(path).parts:
        part = part.lower()
        for label, keys in LABEL_KEYWORDS.items():
            for k in keys:
                if k in part:
                    return label
    return None

# -----------------------
# Dataset discovery
# -----------------------
def discover_images_and_labels(root: Path):
    print(f"Scanning for images under {root} ...")
    image_paths = []
    labels = []
    for ext in IMAGE_EXTS:
        for p in root.rglob(f"*{ext}"):
            image_paths.append(str(p))
            labels.append(infer_label_from_path(p))

    # Filter out images with no inferred label (but keep track)
    kept_paths = []
    kept_labels = []
    unlabeled = []
    for p, l in zip(image_paths, labels):
        if l is None:
            unlabeled.append(p)
        else:
            kept_paths.append(p)
            kept_labels.append(l)

    print(f"Found {len(image_paths)} images, {len(kept_paths)} labeled, {len(unlabeled)} unlabeled (sample below)")
    if unlabeled:
        for s in unlabeled[:10]:
            print("  unlabeled:", s)
    return kept_paths, kept_labels

# -----------------------
# Build class mapping dynamically
# -----------------------
def build_class_mapping(labels_list):
    unique = sorted(list(set(labels_list)))
    mapping = {name: idx for idx, name in enumerate(unique)}
    inv_map = {v:k for k,v in mapping.items()}
    return mapping, inv_map

# -----------------------
# Dataset class
# -----------------------
class RetinalDataset(Dataset):
    def __init__(self, image_paths, labels, class_map, transform=None):
        assert len(image_paths) == len(labels)
        self.image_paths = image_paths
        # convert string labels to int
        self.labels = [class_map[l] for l in labels]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Warning: could not open {img_path}: {e}. Skipping by returning a black image.")
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))

        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# -----------------------
# Train function
# -----------------------
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    image_paths, string_labels = discover_images_and_labels(RAW_ROOT)
    if len(image_paths) == 0:
        raise RuntimeError(f"No labeled images discovered under {RAW_ROOT}. Check your folder structure or label keywords.")

    class_map, inv_map = build_class_mapping(string_labels)
    print("Class mapping:", class_map)
    NUM_CLASSES = len(class_map)
    print(f"Number of classes inferred: {NUM_CLASSES}")

    # Show distribution
    counts = Counter(string_labels)
    print("Class distribution (string labels):")
    for lbl, c in counts.items():
        print(f"  {lbl}: {c}")

    # Create dataset splits
    dataset_size = len(image_paths)
    indices = list(range(dataset_size))
    random.shuffle(indices)

    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size+val_size]
    test_idx = indices[train_size+val_size:]

    train_paths = [image_paths[i] for i in train_idx]
    train_labels = [string_labels[i] for i in train_idx]
    val_paths = [image_paths[i] for i in val_idx]
    val_labels = [string_labels[i] for i in val_idx]
    test_paths = [image_paths[i] for i in test_idx]
    test_labels = [string_labels[i] for i in test_idx]

    train_ds = RetinalDataset(train_paths, train_labels, class_map, transform=train_transform)
    val_ds = RetinalDataset(val_paths, val_labels, class_map, transform=val_transform)
    test_ds = RetinalDataset(test_paths, test_labels, class_map, transform=val_transform)

    num_workers = min(8, os.cpu_count() or 0)
    pin_memory = True if device.type == "cuda" else False

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    print(f"Splits -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Initialize model: use dynamic num classes
    model = GlaucomaModel(num_classes=NUM_CLASSES, model_name=MODEL_NAME)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        for imgs, targets in loop:
            imgs = imgs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)  # expect shape (N, NUM_CLASSES)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(targets).sum().item()
            total += targets.size(0)
            loop.set_postfix(loss=loss.item(), acc=f"{100.*correct/total:.2f}%")

        epoch_train_loss = running_loss / len(train_ds)
        epoch_train_acc = 100. * correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # Validation
        model.eval()
        v_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]")
            for imgs, targets in loop:
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                v_loss += loss.item() * imgs.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(targets).sum().item()
                total += targets.size(0)
                loop.set_postfix(loss=loss.item(), acc=f"{100.*correct/total:.2f}%")

        epoch_val_loss = v_loss / len(val_ds)
        epoch_val_acc = 100. * correct / total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        scheduler.step(epoch_val_loss)
        print(f"Epoch {epoch+1}: Train loss {epoch_train_loss:.4f}, Train acc {epoch_train_acc:.2f} | Val loss {epoch_val_loss:.4f}, Val acc {epoch_val_acc:.2f}")

        # Save best
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "combined_model.pth"))
            print("Saved best model to", os.path.join(OUTPUT_DIR, "combined_model.pth"))

    # final save
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "final_combined_model.pth"))
    print("Saved final model to", os.path.join(OUTPUT_DIR, "final_combined_model.pth"))

    # optional safetensors
    try:
        import safetensors.torch
        safetensors.torch.save_file(model.state_dict(), os.path.join(OUTPUT_DIR, "combined_model.safetensors"))
        print("Saved safetensors.")
    except Exception:
        print("safetensors not installed or failed -> skipped.")

    # Plot history
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.title("Loss")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(train_accs, label="train")
    plt.plot(val_accs, label="val")
    plt.title("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "training_history.png"))

    # Evaluate on test set with per-class accuracy
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0]*NUM_CLASSES
    class_total = [0]*NUM_CLASSES

    with torch.no_grad():
        for imgs, targets in tqdm(test_loader, desc="Testing"):
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            total += targets.size(0)
            correct += preds.eq(targets).sum().item()
            for i in range(targets.size(0)):
                l = targets[i].item()
                class_total[l] += 1
                if preds[i].item() == l:
                    class_correct[l] += 1

    test_loss = test_loss / len(test_ds)
    test_acc = 100. * correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    inv_map = {v:k for k,v in class_map.items()}
    for cls_idx in range(NUM_CLASSES):
        if class_total[cls_idx] > 0:
            print(f"  {inv_map[cls_idx]}: {100.*class_correct[cls_idx]/class_total[cls_idx]:.2f}% ({class_total[cls_idx]} samples)")
        else:
            print(f"  {inv_map[cls_idx]}: N/A (no samples)")

if __name__ == "__main__":
    train_model()
