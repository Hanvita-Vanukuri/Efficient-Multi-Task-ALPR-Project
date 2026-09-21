from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (
    CCPDDataset,
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

CHECKPOINT = Path(
    "checkpoints/best_model.pth"
)


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
# GET LOGITS
# ============================================================

def get_recognition_logits(output):

    if isinstance(output, dict):

        if "recognition" not in output:
            raise KeyError(
                "Model output does not contain "
                "'recognition'."
            )

        return output["recognition"]

    if torch.is_tensor(output):
        return output

    raise TypeError(
        f"Unsupported model output: {type(output)}"
    )


# ============================================================
# CTC DECODER
# ============================================================

def decode_ctc(logits):

    predictions = logits.argmax(dim=-1)

    decoded = []

    for row in predictions:

        result = []
        previous = None

        for index in row.tolist():

            # Blank
            if index == BLANK_IDX:
                previous = index
                continue

            # Repeated CTC character
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

    if len(target) == 0:

        return (
            1.0
            if len(predicted) == 0
            else 0.0
        )

    correct = 0
    total = max(
        len(predicted),
        len(target)
    )

    for i in range(
        min(len(predicted), len(target))
    ):

        if predicted[i] == target[i]:
            correct += 1

    return correct / total


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("ALPR FULL TEST-SET EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Basic checks
    # --------------------------------------------------------

    print()
    print("Test file   :", TEST_FILE)
    print("Checkpoint  :", CHECKPOINT)
    print("Characters  :", len(CHARS))
    print("CTC classes :", NUM_CLASSES)
    print("Blank index :", BLANK_IDX)

    if not TEST_FILE.exists():

        raise FileNotFoundError(
            f"Test split not found:\n{TEST_FILE}"
        )

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n{CHECKPOINT}\n\n"
            "Train the model first."
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = select_device()

    print()
    print("Device:", device)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = CCPDDataset(
        TEST_FILE,
        transform=transform
    )

    print(
        "Test images:",
        len(dataset)
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=ctc_collate_fn
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = LightEdgeALPR(
        num_classes=NUM_CLASSES
    )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    print()
    print("Loading checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device
    )

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        saved_epoch = checkpoint.get(
            "epoch",
            "unknown"
        )

        saved_val_loss = checkpoint.get(
            "val_loss",
            "unknown"
        )

    else:

        # Support raw state_dict checkpoint
        model.load_state_dict(
            checkpoint
        )

        saved_epoch = "unknown"
        saved_val_loss = "unknown"

    model = model.to(device)

    model.eval()

    print(
        "Checkpoint epoch:",
        saved_epoch
    )

    print(
        "Checkpoint val loss:",
        saved_val_loss
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    total_samples = 0

    exact_matches = 0

    total_character_score = 0.0

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print()
    print("Evaluating complete test set...")
    print()

    with torch.inference_mode():

        for batch_index, (
            images,
            targets,
            target_lengths,
            plate_numbers
        ) in enumerate(loader):

            images = images.to(device)

            output = model(images)

            logits = get_recognition_logits(output)

            predictions = decode_ctc(logits)

            for predicted, actual in zip(
                predictions,
                plate_numbers
            ):

                total_samples += 1

                # Full plate exact match
                if predicted == actual:
                    exact_matches += 1

                # Character-level score
                total_character_score += (
                    character_accuracy(
                        predicted,
                        actual
                    )
                )

            if (
                batch_index + 1
            ) % 100 == 0:

                current_accuracy = (
                    100.0
                    * exact_matches
                    / total_samples
                )

                print(
                    f"Processed "
                    f"{total_samples} / {len(dataset)} | "
                    f"Current Plate Accuracy: "
                    f"{current_accuracy:.2f}%"
                )

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    full_plate_accuracy = (
        100.0
        * exact_matches
        / total_samples
        if total_samples > 0
        else 0.0
    )

    average_character_accuracy = (
        100.0
        * total_character_score
        / total_samples
        if total_samples > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL ALPR RESULTS")
    print("=" * 70)

    print()
    print(
        "Total test images       :",
        total_samples
    )

    print(
        "Completely correct plates:",
        exact_matches
    )

    print(
        "Incorrect plates         :",
        total_samples - exact_matches
    )

    print()
    print(
        f"FULL PLATE ACCURACY     : "
        f"{full_plate_accuracy:.2f}%"
    )

    print(
        f"CHARACTER ACCURACY     : "
        f"{average_character_accuracy:.2f}%"
    )

    print()
    print("=" * 70)
    print("Evaluation completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()