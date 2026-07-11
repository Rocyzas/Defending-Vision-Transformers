import argparse
import csv
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import PixelAttack, ThresholdAttack, SquareAttack

from ignite.metrics import SSIM, PSNR

from greedypixel.greedypixel import GreedyPixel

from src.model import *

import warnings
warnings.filterwarnings("ignore", message="function_values is not a list of scalars")


# ImageNet statistics the model was trained with (see Data_Augmented_Training_hybrid.py)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class NormalizedModel(nn.Module):
    """Wraps a model so it accepts inputs in [0, 1] pixel space and applies
    ImageNet normalization internally.

    This keeps the adversarial search space in [0, 1], which is what
    ART's clip_values=(0, 1) and GreedyPixel's internal clamp(0, 1) assume.
    """

    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ART evasion attacks (PixelAttack, ThresholdAttack) on a PyTorch model "
                    "using images from a test folder (ImageFolder-style: one subfolder per class)."
    )
    parser.add_argument("model_path", type=str, help="Path to the PyTorch model file (.pt)")
    parser.add_argument("test_dir", type=str,
                         help="Path to test data folder, structured as <test_dir>/<class_name>/*.png")
    parser.add_argument("--input-shape", type=int, nargs=3, default=(3, 96, 96),
                         metavar=("C", "H", "W"), help="Input shape (default: 3 96 96)")
    parser.add_argument("--nb-classes", type=int, default=4, help="Number of classes (default: 4)")
    parser.add_argument("--clip-min", type=float, default=0.0, help="Min pixel value (default: 0.0)")
    parser.add_argument("--clip-max", type=float, default=1.0, help="Max pixel value (default: 1.0)")
    parser.add_argument("--max-iter", type=int, default=300, help="Max iterations for attacks (default: 100)")
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
    # NOTE: normalization is applied *inside* the model (NormalizedModel), not here,
    # so the attack operates in [0, 1] pixel space (matching clip_values=(0, 1)).
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


def compute_perturbation_metrics(x_orig, x_adv, device, data_range=1.0):
    """Average SSIM and PSNR of the adversarial images against the originals.

    Both arrays are (N, C, H, W) in the clip range (default [0, 1]); the caller
    may pass a filtered subset (e.g. only successfully-attacked images). The
    adversarial image plays the role of ``y_pred`` and the original image the
    role of ``y`` for the ignite metrics; the result is averaged over the batch.
    """
    orig = torch.from_numpy(np.ascontiguousarray(x_orig)).float().to(device)
    adv = torch.from_numpy(np.ascontiguousarray(x_adv)).float().to(device)

    ssim_metric = SSIM(data_range=data_range, device=device)
    psnr_metric = PSNR(data_range=data_range, device=device)

    ssim_metric.update((adv, orig))
    psnr_metric.update((adv, orig))

    return float(ssim_metric.compute()), float(psnr_metric.compute())


def main():
    args = parse_args()

    # ---- Device setup ----
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"Using device: {device}")

    # ---- Load model ----
    print('-->', os.getcwd(), '<--')
    
    
    state = torch.load(args.model_path, map_location=device, weights_only=True)
    base_model = build_vit_base_r50_s16_224(num_classes=4, img_size=96)
    base_model.load_state_dict(state)

    # Wrap so the model normalizes internally and accepts [0, 1] inputs.
    model = NormalizedModel(base_model, IMAGENET_MEAN, IMAGENET_STD).to(device)
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
        device_type="cuda" if device.type == "cuda" else "mps",
    )

    # ---- Load test data from folder ----
    x, y = load_test_folder(args.test_dir, input_shape, args.batch_size, args.max_samples)
    n_samples = x.shape[0]

    # ---- Define attacks ----
    attacks = [
        SquareAttack(estimator=classifier, eps=4/255, max_iter=args.max_iter, norm="inf", nb_restarts=1, verbose=True),
        # es=1 for DE algorithm
        # PixelAttack(classifier=classifier, th=1, es=1, max_iter=10, targeted=False, verbose=True),
        # ThresholdAttack(classifier=classifier, th=None, es=0, max_iter=args.max_iter, targeted=False, verbose=True),
        GreedyPixel(
            target = model,          # raw torch module: called directly, returns logit tensors
            surrogate = None,
            eps = 8 / 255,          # L-inf budget in [0, 1] pixel space
            # max_query = args.max_query,
            early_stop = True,
            # batch_size = 128
            )
    ]

    # ---- Run attacks ----
    preds_clean = np.argmax(classifier.predict(x), axis=1)
    clean_acc = np.mean(preds_clean == y)
    print(f"Clean accuracy: {clean_acc:.4f}")

    results = []
    timestamp = datetime.now().isoformat(timespec="seconds")

    for attack in attacks:
        print(f"\nRunning {os.name}...")
        name = attack.__class__.__name__

        start_time = time.perf_counter()
        if name == "GreedyPixel":
            # GreedyPixel works on one image at a time and expects torch tensors.
            x_t = torch.from_numpy(x).float().to(device)
            y_t = torch.from_numpy(y).long().to(device)
            adv_batches = []
            for i in range(x_t.shape[0]):
                x_adv_i, _ = attack.attack(x_t[i:i + 1], y_t[i:i + 1])
                adv_batches.append(x_adv_i.cpu())
            x_adv = torch.cat(adv_batches, dim=0).numpy().astype(np.float32)

        else:
            x_adv = attack.generate(x=x, y=y)
        attack_time = time.perf_counter() - start_time
        print(f"{name}: attack time = {attack_time:.2f} s")

        preds_adv = np.argmax(classifier.predict(x_adv), axis=1)
        adv_acc = np.mean(preds_adv == y)

        # classifier accuracy after adversarial attack
        print(f"{name}: adversarial accuracy = {adv_acc:.4f}")

        # Standard untargeted ASR: among samples classified correctly on clean
        # input, the fraction the attack flips to incorrect. (Equals 1 - adv_acc
        # only when clean accuracy is 100%.)
        clean_correct = preds_clean == y
        n_clean_correct = int(clean_correct.sum())
        if n_clean_correct > 0:
            asr = float(np.mean(preds_adv[clean_correct] != y[clean_correct]))
        else:
            asr = float("nan")
        print(f"{name}: ASR = {asr:.4f}")

        # Perceptual similarity of the SUCCESSFUL adversarial examples vs. the
        # originals. Successful = originally correct and now flipped (the same
        # subset as the ASR numerator); failed/unperturbed images would otherwise
        # inflate SSIM toward 1 and PSNR toward high values.
        success_mask = clean_correct & (preds_adv != y)
        n_success = int(success_mask.sum())
        if n_success > 0:
            ssim_val, psnr_val = compute_perturbation_metrics(
                x[success_mask], x_adv[success_mask], device,
                data_range=(args.clip_max - args.clip_min),
            )
        else:
            ssim_val, psnr_val = float("nan"), float("nan")
        print(f"{name}: SSIM = {ssim_val:.4f}, PSNR = {psnr_val:.4f} dB "
              f"(over {n_success} successful attacks)")

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
            "asr": round(asr, 4),
            "ssim": round(ssim_val, 4),
            "psnr": round(psnr_val, 4),
            "attack_time_s": round(attack_time, 2),
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