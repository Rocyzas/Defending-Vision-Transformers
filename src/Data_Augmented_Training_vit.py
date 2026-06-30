import os
# Work around duplicate OpenMP runtime (conda numpy + pip torch both ship libomp.dylib).
# Must be set BEFORE importing torch/numpy, otherwise the process aborts (OMP: Error #15).
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import argparse
# Run from repo root or src/; add repo root so `from src.model import ...` resolves.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassF1Score
from PIL import Image
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from torchvision import transforms
from src.model import *
from src.augmentations import AUGMENTATIONS
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy, MulticlassPrecision, MulticlassRecall


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")


class ImageDataset(Dataset):
    def __init__(self, csv_path, split='train', transform=None):

        self.transform = transform

        full_df = pd.read_csv(csv_path)

        self.classes = sorted(full_df['label'].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.df = full_df[full_df['split'] == split].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No samples found for split: {split}. Check your CSV or directory paths.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Read the stored absolute path and trim it to a repo-relative path
        # (keep the last 4 components: data/<split-dir>/<class>/<file>).
        abs_path = row['abs_path']
        # img_path = '/'.join(abs_path.split('/')[-4:])
        img_path = '/'.join(abs_path.split('/')[-3:])
        # img_path = '../' + img_path
        label_name = row['label']

        label = self.class_to_idx[label_name]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise e

        if self.transform:
            image = self.transform(image)

        return image, label


# Data Setup

image_size = 96

pre_norm = [
    transforms.Resize((image_size, image_size)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    # augmentation goes here (on [0,1] tensor, before normalization)
]

post_norm = [
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]

base_pipeline = pre_norm + post_norm

parser = argparse.ArgumentParser(description="Augmented ViT training (hybrid).")
parser.add_argument('--data', required=True,
                    help="Path to the dataset CSV file.")
parser.add_argument('--models', required=True,
                    help="Directory where trained models are saved.")
args = parser.parse_args()

csv_path = args.data
models_dir = args.models
os.makedirs(models_dir, exist_ok=True)


# Training Loop (one model per augmentation)

num_epochs = 25
aug_p = 0.1

for aug_name, aug_factory in AUGMENTATIONS.items():
    print(f"\n{'='*60}")
    print(f"Training with augmentation: {aug_name}")
    print(f"{'='*60}\n")

    # Build train pipeline: augment on [0,1] tensor, then normalize
    train_pipeline = pre_norm + [transforms.RandomApply([aug_factory()], p=aug_p)] + post_norm

    train_dataset = ImageDataset(csv_path=csv_path, split='train', transform=transforms.Compose(train_pipeline))
    val_dataset   = ImageDataset(csv_path=csv_path, split='val',   transform=transforms.Compose(base_pipeline))
    test_dataset  = ImageDataset(csv_path=csv_path, split='test',  transform=transforms.Compose(base_pipeline))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

    num_classes = len(train_dataset.classes)
    model = build_vit_base_patch16_224(num_classes=num_classes, img_size=image_size).to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=0.1, patience=2, min_lr=1e-6
    )

    f1_metric        = MulticlassF1Score(num_classes=num_classes, average=None).to(device)
    macro_f1_metric  = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)
    accuracy_metric  = MulticlassAccuracy(num_classes=num_classes, average='macro').to(device)
    precision_metric = MulticlassPrecision(num_classes=num_classes, average='macro').to(device)
    recall_metric    = MulticlassRecall(num_classes=num_classes, average='macro').to(device)

    early_stop_patience = 5
    early_stop_counter  = 0
    best_val_loss       = float('inf')

    log_time = datetime.now().strftime("%m-%d_%H%M")
    writer = SummaryWriter(f'runs/ViT_{log_time}_96x96_{aug_name}')

    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        macro_f1_metric.reset()
        train_loss = 0.0

        train_loop = tqdm(train_loader, leave=False)
        for img, labels in train_loop:
            img, labels = img.to(device), labels.to(device)
            optimiser.zero_grad()
            output = model(img)
            loss = criterion(output, labels)
            loss.backward()
            optimiser.step()
            train_loss += loss.item()
            macro_f1_metric.update(output, labels)
            train_loop.set_description(f"[{aug_name}] Epoch [{epoch+1}/{num_epochs}]")

        avg_train_loss = train_loss / len(train_loader)
        epoch_train_f1 = macro_f1_metric.compute().item()

        # --- Validation ---
        model.eval()
        f1_metric.reset()
        macro_f1_metric.reset()
        accuracy_metric.reset()
        precision_metric.reset()
        recall_metric.reset()

        val_loss = 0.0
        with torch.no_grad():
            for img, labels in val_loader:
                img, labels = img.to(device), labels.to(device)
                output = model(img)
                val_loss += criterion(output, labels).item()
                f1_metric.update(output, labels)
                macro_f1_metric.update(output, labels)
                accuracy_metric.update(output, labels)
                precision_metric.update(output, labels)
                recall_metric.update(output, labels)

        avg_val_loss = val_loss / len(val_loader)
        val_f1  = macro_f1_metric.compute().item()
        val_acc = accuracy_metric.compute().item()
        val_prec = precision_metric.compute().item()
        val_rec  = recall_metric.compute().item()

        writer.add_scalar('Loss/Train',           avg_train_loss, epoch)
        writer.add_scalar('Loss/Validation',      avg_val_loss,   epoch)
        writer.add_scalar('Accuracy/Validation',  val_acc,        epoch)
        writer.add_scalar('Precision/Validation', val_prec,       epoch)
        writer.add_scalar('Recall/Validation',    val_rec,        epoch)
        writer.add_scalar('F1/Validation',        val_f1,         epoch)

        print(f"Epoch {epoch+1}: Loss(V): {avg_val_loss:.4f} | Acc(V): {val_acc:.4f} | F1(V): {val_f1:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(models_dir, f'best_vit_model_96x96_{aug_name}_{aug_p}.pth'))
            early_stop_counter = 0
            print("--> Model Saved!")
        else:
            early_stop_counter += 1
            print(f"--> EarlyStopping Counter: {early_stop_counter}/{early_stop_patience}")

        if early_stop_counter >= early_stop_patience:
            print("!!! Early Stopping Triggered !!!")
            break

    writer.close()
    del model, optimiser, criterion, scheduler
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"\nFinished: {aug_name}")
