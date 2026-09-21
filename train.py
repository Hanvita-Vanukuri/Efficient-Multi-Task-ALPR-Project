from pathlib import Path
import csv
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (
    CCPDDataset,
    TRAIN_FILE,
    TEST_FILE,
    CHARS,
    NUM_CLASSES,
    BLANK_IDX,
    transform,
    ctc_collate_fn
)

from model import LightEdgeALPR


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 16
NUM_EPOCHS = 30

LEARNING_RATE = 0.001

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 160

NUM_WORKERS = 0

EARLY_STOPPING_PATIENCE = 5

CHECKPOINT_DIR = Path("checkpoints")
RESULTS_DIR = Path("results")

BEST_CHECKPOINT = CHECKPOINT_DIR / "best_model.pth"
LAST_CHECKPOINT = CHECKPOINT_DIR / "last_model.pth"

HISTORY_FILE = RESULTS_DIR / "training_history.csv"


# ============================================================
# DEVICE
# ============================================================

def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# MODEL OUTPUT HELPER
# ============================================================

def get_recognition_logits(output):
    """
    Supports:
        Tensor
        {"recognition": Tensor}
    """

    if isinstance(output, dict):
        if "recognition" not in output:
            raise KeyError(
                "Model output dictionary does not contain "
                "'recognition'."
            )

        return output["recognition"]

    if torch.is_tensor(output):
        return output

    raise TypeError(
        f"Unsupported model output type: {type(output)}"
    )


# ============================================================
# CTC GREEDY DECODER
# ============================================================

def decode_ctc(logits):
    """
    Greedy CTC decoder.

    Input:
        [B, T, C]

    Output:
        list of decoded strings
    """

    predictions = logits.argmax(dim=-1)

    decoded = []

    for row in predictions:
        result = []
        previous = None

        for index in row.tolist():

            # CTC blank
            if index == BLANK_IDX:
                previous = index
                continue

            # Remove repeated consecutive characters
            if index == previous:
                continue

            if 0 <= index < len(CHARS):
                result.append(CHARS[index])

            previous = index

        decoded.append("".join(result))

    return decoded


# ============================================================
# CHARACTER ACCURACY
# ============================================================

def character_accuracy(predicted, target):
    """
    Position-wise character accuracy.
    """

    if len(target) == 0:
        return 1.0 if len(predicted) == 0 else 0.0

    correct = 0
    total = max(len(predicted), len(target))

    for i in range(min(len(predicted), len(target))):
        if predicted[i] == target[i]:
            correct += 1

    return correct / total


# ============================================================
# TRY CTC ON DEVICE
# ============================================================

def ctc_supported_on_device(device):
    try:
        test_ctc = nn.CTCLoss(
            blank=BLANK_IDX,
            zero_infinity=True
        )

        # Small test tensors
        T = 5
        N = 2
        C = NUM_CLASSES

        test_logits = torch.randn(
            T,
            N,
            C,
            device=device,
            requires_grad=True
        )

        targets = torch.tensor(
            [1, 2, 3, 4],
            dtype=torch.long,
            device=device
        )

        input_lengths = torch.full(
            (N,),
            T,
            dtype=torch.long,
            device=device
        )

        target_lengths = torch.tensor(
            [2, 2],
            dtype=torch.long,
            device=device
        )

        loss = test_ctc(
            test_logits,
            targets,
            input_lengths,
            target_lengths
        )

        loss.backward()

        return True

    except Exception as error:
        print()
        print("CTC test failed on device:")
        print(error)
        return False


# ============================================================
# VALIDATION
# ============================================================

def validate(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0
    total_batches = 0

    total_exact = 0
    total_samples = 0

    total_char_score = 0.0

    with torch.no_grad():

        for images, targets, target_lengths, plate_numbers in loader:

            images = images.to(device)

            output = model(images)
            logits = get_recognition_logits(output)

            # [B,T,C] -> [T,B,C]
            logits_ctc = logits.permute(1, 0, 2)

            batch_size = images.size(0)
            time_steps = logits.size(1)

            input_lengths = torch.full(
                (batch_size,),
                time_steps,
                dtype=torch.long,
                device=logits.device
            )

            target_lengths_device = target_lengths.to(
                logits.device
            )

            targets_device = targets.to(
                logits.device
            )

            loss = criterion(
                logits_ctc,
                targets_device,
                input_lengths,
                target_lengths_device
            )

            total_loss += loss.item()
            total_batches += 1

            # Decode
            predictions = decode_ctc(logits)

            for predicted, target in zip(
                predictions,
                plate_numbers
            ):
                total_samples += 1

                if predicted == target:
                    total_exact += 1

                total_char_score += character_accuracy(
                    predicted,
                    target
                )

    avg_loss = (
        total_loss / total_batches
        if total_batches > 0
        else math.inf
    )

    exact_accuracy = (
        100.0 * total_exact / total_samples
        if total_samples > 0
        else 0.0
    )

    char_accuracy = (
        100.0 * total_char_score / total_samples
        if total_samples > 0
        else 0.0
    )

    return avg_loss, exact_accuracy, char_accuracy


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LIGHT-EDGE ALPR TRAINING")
    print("=" * 70)

    print()
    print("Train file :", TRAIN_FILE)
    print("Test file  :", TEST_FILE)
    print("Characters :", len(CHARS))
    print("CTC classes:", NUM_CLASSES)
    print("Blank index:", BLANK_IDX)

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training split not found:\n{TRAIN_FILE}"
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test split not found:\n{TEST_FILE}"
        )

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    print()
    print("Loading training dataset...")

    train_dataset = CCPDDataset(
        TRAIN_FILE,
        transform=transform
    )

    print()
    print("Loading validation dataset...")

    val_dataset = CCPDDataset(
        TEST_FILE,
        transform=transform
    )

    print()
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        collate_fn=ctc_collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        collate_fn=ctc_collate_fn
    )

    # --------------------------------------------------------
    # Initial device
    # --------------------------------------------------------

    device = select_device()

    print()
    print("Initial device:", device)

    # --------------------------------------------------------
    # CTC compatibility test
    # --------------------------------------------------------

    if not ctc_supported_on_device(device):

        print()
        print("CTC is not working correctly on", device)
        print("Switching training to CPU.")

        device = torch.device("cpu")

    print()
    print("Final training device:", device)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = LightEdgeALPR(
        num_classes=NUM_CLASSES
    )

    model = model.to(device)

    print()
    print("Model created successfully.")

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = nn.CTCLoss(
        blank=BLANK_IDX,
        zero_infinity=True
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Learning-rate scheduler
    # Reduce every 10 epochs
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.1
    )

    # --------------------------------------------------------
    # Create folders
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Start fresh
    # --------------------------------------------------------

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    history = []

    print()
    print("Starting training...")
    print()

    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------

    for epoch in range(1, NUM_EPOCHS + 1):

        model.train()

        running_loss = 0.0
        batch_count = 0

        for images, targets, target_lengths, _ in train_loader:

            images = images.to(device)

            targets = targets.to(device)

            target_lengths = target_lengths.to(device)

            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            output = model(images)

            logits = get_recognition_logits(output)

            # ------------------------------------------------
            # CTC expects [T,B,C]
            # ------------------------------------------------

            logits_ctc = logits.permute(1, 0, 2)

            batch_size = images.size(0)
            time_steps = logits.size(1)

            input_lengths = torch.full(
                (batch_size,),
                time_steps,
                dtype=torch.long,
                device=device
            )

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss = criterion(
                logits_ctc,
                targets,
                input_lengths,
                target_lengths
            )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            optimizer.zero_grad()

            loss.backward()

            # Gradient clipping helps stability
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

            running_loss += loss.item()
            batch_count += 1

        train_loss = (
            running_loss / batch_count
            if batch_count > 0
            else float("inf")
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_loss, exact_accuracy, char_accuracy = validate(
            model,
            val_loader,
            criterion,
            device
        )

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Plate Accuracy: {exact_accuracy:.2f}% | "
            f"Char Accuracy: {char_accuracy:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        # ----------------------------------------------------
        # Save history
        # ----------------------------------------------------

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "plate_accuracy": exact_accuracy,
            "character_accuracy": char_accuracy,
            "learning_rate": current_lr
        })

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "plate_accuracy": exact_accuracy,
                "character_accuracy": char_accuracy,
                "chars": CHARS,
                "num_classes": NUM_CLASSES,
                "blank_idx": BLANK_IDX
            }

            torch.save(
                checkpoint,
                BEST_CHECKPOINT
            )

            print(
                f"  -> Best model saved: "
                f"{BEST_CHECKPOINT}"
            )

        else:
            patience_counter += 1

        # ----------------------------------------------------
        # Save last model
        # ----------------------------------------------------

        last_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "plate_accuracy": exact_accuracy,
            "character_accuracy": char_accuracy,
            "chars": CHARS,
            "num_classes": NUM_CLASSES,
            "blank_idx": BLANK_IDX
        }

        torch.save(
            last_checkpoint,
            LAST_CHECKPOINT
        )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step()

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if patience_counter >= EARLY_STOPPING_PATIENCE:

            print()
            print(
                "Early stopping triggered."
            )

            break

    # --------------------------------------------------------
    # Write CSV
    # --------------------------------------------------------

    with open(
        HISTORY_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "plate_accuracy",
                "character_accuracy",
                "learning_rate"
            ]
        )

        writer.writeheader()
        writer.writerows(history)

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)

    print("Best epoch       :", best_epoch)
    print("Best validation loss:", best_val_loss)

    print()
    print("Best model:")
    print(BEST_CHECKPOINT)

    print()
    print("Last model:")
    print(LAST_CHECKPOINT)

    print()
    print("Training history:")
    print(HISTORY_FILE)


if __name__ == "__main__":
    main()