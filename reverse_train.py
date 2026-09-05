"""Fine-tune a high-ASR backdoored model using true-labelled data.

The training loader returns clean images. Inside the training loop, an exact
configurable fraction of each batch receives the trigger while the remainder
stays clean. Every image retains its ground-truth label. Clean test accuracy and
ASR are evaluated after each epoch. Once ASR reaches the requested threshold,
a clean-only recovery stage improves clean accuracy while retaining only
the recovery checkpoint with the highest clean accuracy. ASR is still measured
and logged so the clean-accuracy/backdoor tradeoff remains visible.
"""

import argparse
import logging
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import masks
from logging_config import configure_logging
from models.resnet import ResNet18
from poison import poison


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

logger = logging.getLogger(__name__)


class TriggeredDataset(torch.utils.data.Dataset):
    """Apply a trigger and return the image's ground-truth label."""

    def __init__(self, cifar10, mask, pattern, transform, indexes=None):
        self.cifar10 = cifar10
        self.mask = mask.cpu()
        self.pattern = pattern.cpu()
        self.transform = transform
        self.indexes = (
            list(range(len(cifar10))) if indexes is None else list(indexes)
        )

    def __getitem__(self, index):
        image, true_label = self.cifar10[self.indexes[index]]
        image = poison(image, self.mask, self.pattern)
        return self.transform(image), true_label

    def __len__(self):
        return len(self.indexes)


def _load_state_dict(model, checkpoint_file, device):
    checkpoint = torch.load(checkpoint_file, map_location=device)

    if isinstance(checkpoint, nn.Module):
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Accept checkpoints saved both with and without torch DataParallel.
    model_uses_module = next(iter(model.state_dict())).startswith("module.")
    file_uses_module = next(iter(state_dict)).startswith("module.")

    if file_uses_module and not model_uses_module:
        state_dict = {
            key.removeprefix("module."): value
            for key, value in state_dict.items()
        }
    elif model_uses_module and not file_uses_module:
        state_dict = {
            "module." + key: value for key, value in state_dict.items()
        }

    model.load_state_dict(state_dict)


def attack_success_rate(model, loader, target_class, device):
    """Return the percentage of non-target images classified as the target."""

    model.eval()
    successes = 0
    total = 0

    with torch.no_grad():
        for inputs, true_labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            true_labels = true_labels.to(device, non_blocking=True)

            keep = true_labels.ne(target_class)
            if not keep.any():
                continue

            predictions = model(inputs[keep]).argmax(dim=1)
            successes += predictions.eq(target_class).sum().item()
            total += keep.sum().item()

    return 100.0 * successes / total if total else 0.0


def classification_accuracy(model, loader, device):
    """Return clean classification accuracy as a percentage."""

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            predictions = model(inputs).argmax(dim=1)
            correct += predictions.eq(labels).sum().item()
            total += labels.numel()

    return 100.0 * correct / total if total else 0.0


def reverse_train(
    checkpoint_file,
    output_file,
    mask,
    pattern,
    target_class,
    asr_threshold=15.0,
    max_epochs=100,
    lr=1e-5,
    batch_size=128,
    momentum=0.9,
    trigger_fraction=0.3,
    recovery_epochs=10,
    recovery_lr=1e-5,
    seed=0,
    device=None,
):
    """Return final ASR, clean accuracy, and the completed epoch count."""

    if not 0.0 <= asr_threshold <= 100.0:
        raise ValueError("asr_threshold must be between 0 and 100")
    if not 0.0 <= trigger_fraction <= 1.0:
        raise ValueError("trigger_fraction must be between 0 and 1")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if recovery_epochs < 0:
        raise ValueError("recovery_epochs must be at least 0")
    if recovery_lr <= 0.0:
        raise ValueError("recovery_lr must be greater than 0")

    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)

    base_train = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    base_test = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transforms.ToTensor(),
    )
    clean_test_set = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        ),
    )

    asr_set = TriggeredDataset(
        cifar10=base_test,
        mask=mask,
        pattern=pattern,
        transform=normalize,
    )

    use_pin_memory = device.type == "cuda"
    clean_loader = torch.utils.data.DataLoader(
        base_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=use_pin_memory,
        # CIFAR-10 has 50,000 training examples, which leaves a final batch of
        # 80 when batch_size=128. Dropping it keeps every batch exactly 128.
        drop_last=True,
    )
    asr_loader = torch.utils.data.DataLoader(
        asr_set,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=use_pin_memory,
    )
    clean_test_loader = torch.utils.data.DataLoader(
        clean_test_set,
        batch_size=256,
        shuffle=False,
        num_workers=2,
        pin_memory=use_pin_memory,
    )

    model = ResNet18().to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    _load_state_dict(model, checkpoint_file, device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # The clean loader creates CPU tensors. Move the trigger tensors once so
    # the trigger can be applied to each batch on the selected device.
    training_mask = mask.to(device)
    training_pattern = pattern.to(device)

    triggered_per_batch = round(batch_size * trigger_fraction)
    clean_per_batch = batch_size - triggered_per_batch
    initial_asr = attack_success_rate(
        model, asr_loader, target_class, device
    )
    initial_clean_accuracy = classification_accuracy(
        model, clean_test_loader, device
    )

    logger.info(
        "Starting reverse training: checkpoint=%s output=%s "
        "target_class=%d asr_threshold=%.2f%% max_epochs=%d lr=%g "
        "batch_size=%d trigger_fraction=%.4f triggered_per_batch=%d "
        "clean_per_batch=%d recovery_epochs=%d recovery_lr=%g device=%s",
        checkpoint_file,
        output_file,
        target_class,
        asr_threshold,
        max_epochs,
        lr,
        batch_size,
        trigger_fraction,
        triggered_per_batch,
        clean_per_batch,
        recovery_epochs,
        recovery_lr,
        device,
    )
    logger.info(
        "Initial metrics: ASR=%.2f%% clean_accuracy=%.2f%%",
        initial_asr,
        initial_clean_accuracy,
    )

    final_asr = initial_asr
    final_clean_accuracy = initial_clean_accuracy
    stopped_epoch = 0
    threshold_reached = False

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0

        for clean_inputs, true_labels in clean_loader:
            clean_inputs = clean_inputs.to(device, non_blocking=True)
            true_labels = true_labels.to(device, non_blocking=True)

            # Randomly select an exact number of images to trigger. For a batch
            # of 128 and trigger_fraction=0.3, 38 images are triggered and 90
            # remain clean. The selected positions change for every batch.
            number_triggered = round(len(clean_inputs) * trigger_fraction)
            permutation = torch.randperm(len(clean_inputs), device=device)
            trigger_positions = permutation[:number_triggered]

            inputs = clean_inputs.clone()
            if number_triggered:
                selected_inputs = clean_inputs.index_select(
                    0, trigger_positions
                )
                triggered_images = torch.stack(
                    [
                        poison(
                            image,
                            training_mask,
                            training_pattern,
                        )
                        for image in selected_inputs
                    ]
                )
                inputs[trigger_positions] = triggered_images

            # Normalize both clean and triggered images after applying the
            # trigger. All images keep their original ground-truth labels.
            inputs = normalize(inputs)
            labels = true_labels

            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        final_asr = attack_success_rate(
            model, asr_loader, target_class, device
        )
        final_clean_accuracy = classification_accuracy(
            model, clean_test_loader, device
        )
        mean_loss = running_loss / len(clean_loader)

        logger.info(
            "Reverse epoch %d/%d: loss=%.4f ASR=%.2f%% "
            "clean_accuracy=%.2f%%",
            epoch,
            max_epochs,
            mean_loss,
            final_asr,
            final_clean_accuracy,
        )

        stopped_epoch = epoch
        if final_asr <= asr_threshold:
            logger.info(
                "ASR threshold reached at reverse epoch %d",
                epoch,
            )
            threshold_reached = True
            break

    if not threshold_reached:
        logger.warning(
            "ASR stayed above %.2f%% after %d epochs; no checkpoint was saved",
            asr_threshold,
            max_epochs,
        )
        return final_asr, final_clean_accuracy, stopped_epoch

    # The first checkpoint satisfying the ASR requirement is the recovery
    # baseline. Later clean-only epochs replace it whenever clean accuracy
    # improves. ASR is recorded but does not block recovery checkpoint saving.
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    torch.save(model.state_dict(), output_file)
    best_asr = final_asr
    best_clean_accuracy = final_clean_accuracy
    best_recovery_epoch = 0
    logger.info(
        "Saved recovery baseline: ASR=%.2f%% clean_accuracy=%.2f%% path=%s",
        best_asr,
        best_clean_accuracy,
        output_file,
    )

    recovery_optimizer = optim.SGD(
        model.parameters(), lr=recovery_lr, momentum=momentum
    )

    for recovery_epoch in range(1, recovery_epochs + 1):
        model.train()
        recovery_loss = 0.0

        for clean_inputs, true_labels in clean_loader:
            clean_inputs = clean_inputs.to(device, non_blocking=True)
            true_labels = true_labels.to(device, non_blocking=True)
            inputs = normalize(clean_inputs)

            recovery_optimizer.zero_grad()
            loss = criterion(model(inputs), true_labels)
            loss.backward()
            recovery_optimizer.step()
            recovery_loss += loss.item()

        final_asr = attack_success_rate(
            model, asr_loader, target_class, device
        )
        final_clean_accuracy = classification_accuracy(
            model, clean_test_loader, device
        )
        mean_recovery_loss = recovery_loss / len(clean_loader)

        logger.info(
            "Clean recovery epoch %d/%d: loss=%.4f ASR=%.2f%% "
            "clean_accuracy=%.2f%%",
            recovery_epoch,
            recovery_epochs,
            mean_recovery_loss,
            final_asr,
            final_clean_accuracy,
        )

        if final_clean_accuracy > best_clean_accuracy:
            best_asr = final_asr
            best_clean_accuracy = final_clean_accuracy
            best_recovery_epoch = recovery_epoch
            torch.save(model.state_dict(), output_file)
            logger.info(
                "Saved improved clean checkpoint at recovery epoch %d: "
                "ASR=%.2f%% clean_accuracy=%.2f%% path=%s",
                recovery_epoch,
                best_asr,
                best_clean_accuracy,
                output_file,
            )

    logger.info(
        "Best clean-accuracy checkpoint: recovery_epoch=%d ASR=%.2f%% "
        "clean_accuracy=%.2f%% path=%s",
        best_recovery_epoch,
        best_asr,
        best_clean_accuracy,
        output_file,
    )
    return best_asr, best_clean_accuracy, stopped_epoch


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune using batches with an exact mixture of clean and "
            "triggered images, all retaining their ground-truth labels."
        )
    )
    parser.add_argument(
        "--backdoor", type=int, choices=range(1, 11), required=True
    )
    parser.add_argument(
        "--checkpoint", required=True, help="High-ASR .pt checkpoint"
    )
    parser.add_argument("--output", help="Output .pt path")
    parser.add_argument("--asr-threshold", type=float, default=12.0)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--trigger-fraction",
        "--trigger_fraction",
        dest="trigger_fraction",
        type=float,
        default=0.3,
        help=(
            "Fraction of each batch that receives the trigger; all labels "
            "remain ground truth (default: 0.3)"
        ),
    )
    parser.add_argument(
        "--recovery-epochs",
        type=int,
        default=10,
        help="Number of clean-only epochs after reaching the ASR target",
    )
    parser.add_argument(
        "--recovery-lr",
        type=float,
        default=1e-5,
        help="Learning rate for clean-only recovery (default: 1e-5)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", help="For example: cuda, cuda:0, or cpu"
    )
    parser.add_argument(
        "--log-file", help="Log file path (default: logs/reverse-*.log)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mask, pattern, name, target_class = getattr(
        masks, f"backdoor{args.backdoor}"
    )()
    output = args.output or f"weights/{name}-reversed.pt"

    configure_logging(
        args.log_file,
        run_name=f"cifar10-reverse-{name}",
    )

    reverse_train(
        checkpoint_file=args.checkpoint,
        output_file=output,
        mask=mask,
        pattern=pattern,
        target_class=target_class,
        asr_threshold=args.asr_threshold,
        max_epochs=args.max_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        momentum=args.momentum,
        trigger_fraction=args.trigger_fraction,
        recovery_epochs=args.recovery_epochs,
        recovery_lr=args.recovery_lr,
        seed=args.seed,
        device=args.device,
    )
