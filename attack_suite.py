import argparse
import csv
import os
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import PixelAttack, ThresholdAttack


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ART evasion attacks (PixelAttack, ThresholdAttack) on a PyTorch model "
                    "using images from a test folder (ImageFolder-style: one subfolder per class)."
    )
    parser.add_argument("model_path", type=str, help="Path to the PyTorch model file (.pt)")
    parser.add_argument("test_dir", type=str,
                         help="Path to test data folder, structured as <test_dir>/<class_name>/*.png")
    parser.add_argument("--input-shape", type=int, nargs=3, default=(3, 32, 32),
                         metavar=("C", "H", "W"), help="Input shape (default: 3 32 32)")
    parser.add_argument("--nb-classes", type=int, default=10, help="Number of classes (default: 10)")
    parser.add_argument("--clip-min", type=float, default=0.0, help="Min pixel value (default: 0.0)")
    parser.add_argument("--clip-max", type=float, default=1.0, help="Max pixel value (default: 1.0)")
    parser.add_argument("--max-iter", type=int, default=100, help="Max iterations for attacks (default: 100)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for loading test data (default: 32)")
    parser.add_argument("--max-samples", type=int, default=None,
                         help="Optional cap on number of test samples to load (attacks are slow)")
    parser.add_argument("--output-csv", type=str, default="attack_results.csv",
                         help="Path to output CSV file (default: attack_results.csv)")
    return parser.parse_args()


def load_test_folder(test_dir, input_shape, batch_size, max_samples=None):
    """Loads images from an ImageFolder-style directory into numpy arrays for ART."""
    channels, height, width = input_shape

    transform_list = [transforms.Resize((height, width))]
    if channels == 1:
        transform_list.append(transforms.Grayscale(num_output_channels=1))
    transform_list.append(transforms.ToTensor())  # scales to [0, 1]
    transform = transforms.Compose(transform_list)

    dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    x_batches, y_batches = [], []
    n_loaded = 0
    for x_batch, y_batch in loader:
        x_batches.append(x_batch.numpy())
        y_batches.append(y_batch.numpy())
        n_loaded += x_batch.shape[0]
        if max_samples is not None and n_loaded >= max_samples:
            break

    x = np.concatenate(x_batches, axis=0).astype(np.float32)
    y = np.concatenate(y_batches, axis=0)

    if max_samples is not None:
        x, y = x[:max_samples], y[:max_samples]

    print(f"Loaded {x.shape[0]} test samples from {test_dir} "
          f"({len(dataset.classes)} classes: {dataset.classes})")
    return x, y


def main():
    args = parse_args()

    # ---- Device setup ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load model ----
    model = torch.load(args.model_path, map_location=device)
    model.to(device)
    model.eval()

    # ---- Config ----
    input_shape = tuple(args.input_shape)
    nb_classes = args.nb_classes
    clip_values = (args.clip_min, args.clip_max)

    loss_fn = nn.CrossEntropyLoss()

    # ---- Wrap model in ART classifier ----
    classifier = PyTorchClassifier(
        model=model,
        loss=loss_fn,
        input_shape=input_shape,
        nb_classes=nb_classes,
        clip_values=clip_values,
        device_type="gpu" if device.type == "cuda" else "cpu",
    )

    # ---- Load test data from folder ----
    x, y = load_test_folder(args.test_dir, input_shape, args.batch_size, args.max_samples)
    n_samples = x.shape[0]

    # ---- Define attacks ----
    attacks = [
        PixelAttack(classifier=classifier, th=None, es=1, max_iter=args.max_iter, targeted=False, verbose=True),
        ThresholdAttack(classifier=classifier, th=None, es=0, max_iter=args.max_iter, targeted=False, verbose=True),
    ]

    # ---- Run attacks ----
    preds_clean = np.argmax(classifier.predict(x), axis=1)
    clean_acc = np.mean(preds_clean == y)
    print(f"Clean accuracy: {clean_acc:.4f}")

    results = []
    timestamp = datetime.now().isoformat(timespec="seconds")

    for attack in attacks:
        name = attack.__class__.__name__
        print(f"\nRunning {name}...")

        x_adv = attack.generate(x=x, y=y)
        preds_adv = np.argmax(classifier.predict(x_adv), axis=1)
        adv_acc = np.mean(preds_adv == y)

        print(f"{name}: adversarial accuracy = {adv_acc:.4f}")

        results.append({
            "timestamp": timestamp,
            "model_path": args.model_path,
            "test_dir": args.test_dir,
            "attack": name,
            "n_samples": n_samples,
            "max_iter": args.max_iter,
            "clean_accuracy": round(float(clean_acc), 4),
            "adversarial_accuracy": round(float(adv_acc), 4),
            "accuracy_drop": round(float(clean_acc - adv_acc), 4),
        })

    # ---- Write results to CSV ----
    fieldnames = list(results[0].keys())
    file_exists = os.path.isfile(args.output_csv)

    with open(args.output_csv, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)

    print(f"\nResults appended to {args.output_csv}")


if __name__ == "__main__":
    main()