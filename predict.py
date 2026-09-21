import json
import random
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model import LightEdgeALPR
from dataset import CHARS


# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = Path("CCPD-Dataset")
CHECKPOINT = Path("checkpoints/best_model.pth")
INDEX_FILE = Path("results/plate_index.json")

IMAGE_WIDTH = 160
IMAGE_HEIGHT = 32
MAX_PLATE_LENGTH = 7
BATCH_SIZE = 16

# Ask for N different dataset plates. The filename contains the
# CCPD ground-truth plate label, so the displayed plate number is exact.
RANDOM_SEED = 42

NUM_CLASSES = len(CHARS) + 1
BLANK_IDX = len(CHARS)

# Display settings
REMOVE_CHINESE_PROVINCE = True


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ============================================================
# CCPD PROVINCE / STATE MAP
# ============================================================

PROVINCE_MAP = {
    "京": "Beijing",
    "津": "Tianjin",
    "冀": "Hebei",
    "晋": "Shanxi",
    "蒙": "Inner Mongolia",
    "辽": "Liaoning",
    "吉": "Jilin",
    "黑": "Heilongjiang",
    "沪": "Shanghai",
    "苏": "Jiangsu",
    "浙": "Zhejiang",
    "皖": "Anhui",
    "闽": "Fujian",
    "赣": "Jiangxi",
    "鲁": "Shandong",
    "豫": "Henan",
    "鄂": "Hubei",
    "湘": "Hunan",
    "粤": "Guangdong",
    "桂": "Guangxi",
    "琼": "Hainan",
    "渝": "Chongqing",
    "川": "Sichuan",
    "贵": "Guizhou",
    "云": "Yunnan",
    "藏": "Tibet",
    "陕": "Shaanxi",
    "甘": "Gansu",
    "青": "Qinghai",
    "宁": "Ningxia",
    "新": "Xinjiang",
    "港": "Hong Kong",
    "澳": "Macau",
    "台": "Taiwan",
}


# ============================================================
# MODEL
# ============================================================

def load_model():
    print()
    print("=" * 60)
    print("                 LOADING ALPR MODEL")
    print("=" * 60)
    print("Device          :", DEVICE)
    print("Characters      :", len(CHARS))
    print("Number of class :", NUM_CLASSES)
    print("Checkpoint      :", CHECKPOINT)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    model = LightEdgeALPR(num_classes=NUM_CLASSES)

    checkpoint = torch.load(CHECKPOINT, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        if "epoch" in checkpoint:
            print("Checkpoint epoch:", checkpoint["epoch"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")
    return model


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))

    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))

    tensor = torch.from_numpy(array).float()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    tensor = (tensor - mean) / std
    return tensor


# ============================================================
# CTC DECODING
# ============================================================

def decode_ctc_single(logits):
    probabilities = F.softmax(logits, dim=2)
    predictions = torch.argmax(probabilities, dim=2)[0]

    decoded = []
    confidence_values = []
    previous = None

    for time_index, index in enumerate(predictions.tolist()):
        if index == BLANK_IDX:
            previous = index
            continue

        if index == previous:
            continue

        if 0 <= index < len(CHARS):
            char = CHARS[index]

            # Do not display Chinese characters in model output.
            if REMOVE_CHINESE_PROVINCE and not char.isascii():
                previous = index
                continue

            decoded.append(char)
            confidence_values.append(
                probabilities[0, time_index, index].item()
            )

        previous = index

        if len(decoded) >= MAX_PLATE_LENGTH:
            break

    plate = "".join(decoded)
    confidence = (
        sum(confidence_values) / len(confidence_values)
        if confidence_values
        else 0.0
    )

    return plate, confidence


def decode_ctc_batch(logits):
    return [
        decode_ctc_single(logits[index:index + 1])
        for index in range(logits.shape[0])
    ]


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_plate(text):
    text = "".join(c for c in text if c.isascii() and c.isalnum())
    return text[:MAX_PLATE_LENGTH].upper()


def normalize_plate(text):
    return clean_plate(text).replace(" ", "")


# ============================================================
# CCPD FILENAME INFORMATION
# ============================================================

def get_province_code_from_filename(filename):
    stem = Path(filename).stem

    if "_" in stem:
        plate_part = stem.split("_", 1)[1]
    else:
        plate_part = stem

    for character in plate_part:
        if character in PROVINCE_MAP:
            return character

    return ""


def get_province_from_filename(filename):
    code = get_province_code_from_filename(filename)
    return PROVINCE_MAP.get(code, "Unknown")


def get_plate_from_filename(filename):
    """
    Extract the exact CCPD plate number from the filename and remove
    the Chinese province character from the displayed result.

    Example:
        66091_皖A33B52.jpg -> A33B52
    """

    stem = Path(filename).stem

    if "_" not in stem:
        return ""

    plate_part = stem.split("_", 1)[1]

    # Keep only ASCII letters/numbers.
    plate_part = "".join(c for c in plate_part if c.isascii() and c.isalnum())
    return clean_plate(plate_part)


# ============================================================
# DATASET INFORMATION
# ============================================================

def get_dataset_images():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

    image_files = [
        path
        for path in DATASET_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and "splits" not in {part.lower() for part in path.parts}
    ]

    return image_files


# ============================================================
# MORE RELIABLE PLATE COLOR DETECTION
# ============================================================

def get_plate_crop(image, image_path):
    """Use CCPD bounding box when available; otherwise use central region."""

    if image is None or image.size == 0:
        return None

    h, w = image.shape[:2]
    stem = Path(image_path).stem

    # Standard CCPD filenames often contain a bbox like -x1&y1_x2&y2-.
    match = re.search(r"-(\d+)&(\d+)_(\d+)&(\d+)-", stem)
    if match:
        x1, y1, x2, y2 = map(int, match.groups())
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        crop = image[y1:y2, x1:x2]
        if crop.size > 0:
            return crop

    # Your simplified filenames do not contain a bbox.
    # Use the central part of the CCPD plate image.
    x1 = int(w * 0.10)
    x2 = int(w * 0.90)
    y1 = int(h * 0.20)
    y2 = int(h * 0.80)

    crop = image[y1:y2, x1:x2]
    return crop if crop.size > 0 else image


def color_scores(crop):
    crop = cv2.resize(crop, (320, 120), interpolation=cv2.INTER_AREA)

    h, w = crop.shape[:2]
    crop = crop[int(0.05 * h):int(0.95 * h), int(0.05 * w):int(0.95 * w)]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    total = float(H.size)

    if total == 0:
        return {}

    # Saturated plate-color masks.
    scores = {
        "Blue": np.count_nonzero(
            (H >= 90) & (H <= 135) & (S >= 80) & (V >= 45)
        ) / total,
        "Yellow": np.count_nonzero(
            (H >= 15) & (H <= 40) & (S >= 80) & (V >= 60)
        ) / total,
        "Green": np.count_nonzero(
            (H >= 35) & (H <= 89) & (S >= 65) & (V >= 45)
        ) / total,
        "Red": np.count_nonzero(
            (((H <= 10) | (H >= 170)) & (S >= 80) & (V >= 45))
        ) / total,
        "White": np.count_nonzero(
            (S <= 55) & (V >= 170)
        ) / total,
        "Black": np.count_nonzero(
            V <= 55
        ) / total,
    }

    return scores


def detect_plate_color(image_path):
    """
    Determine plate color from multiple crops.

    The ccpd_green folder is treated as a strong dataset-level hint because
    CCPD-green specifically contains green/new-energy plates.
    """

    image = cv2.imread(str(image_path))
    if image is None:
        return "Unknown"

    parts = {part.lower() for part in image_path.parts}
    if "ccpd_green" in parts:
        return "Green"

    primary = get_plate_crop(image, image_path)
    crops = []
    if primary is not None and primary.size > 0:
        crops.append(primary)

    h, w = image.shape[:2]
    for x1_ratio, x2_ratio, y1_ratio, y2_ratio in [
        (0.10, 0.90, 0.20, 0.80),
        (0.18, 0.82, 0.25, 0.75),
        (0.25, 0.75, 0.30, 0.70),
    ]:
        crop = image[
            int(h * y1_ratio):int(h * y2_ratio),
            int(w * x1_ratio):int(w * x2_ratio),
        ]
        if crop.size > 0:
            crops.append(crop)

    scores_list = [color_scores(crop) for crop in crops]
    scores_list = [scores for scores in scores_list if scores]

    if not scores_list:
        return "Unknown"

    colors = scores_list[0].keys()
    median_scores = {
        color: float(np.median([scores[color] for scores in scores_list]))
        for color in colors
    }

    ranked = sorted(
        median_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    best_color, best_score = ranked[0]
    second_score = ranked[1][1]

    # Require evidence and separation from the second-best color.
    if best_score < 0.06:
        return "Unknown"

    if best_score < 0.18 and (best_score - second_score) < 0.025:
        return "Unknown"

    return best_color


# ============================================================
# VEHICLE CATEGORY FROM PLATE COLOR
# ============================================================

def classify_vehicle(color):
    mapping = {
        "Blue": "Regular / Private Vehicle",
        "Yellow": "Commercial / Large Vehicle",
        "Green": "New Energy Vehicle (EV)",
        "White": "Police / Special Vehicle",
        "Black": "Foreign / Special Vehicle",
        "Red": "Special / Temporary Vehicle",
        "Unknown": "Unknown",
    }
    return mapping.get(color, "Unknown")


# ============================================================
# SELECT DIVERSE DATASET IMAGES
# ============================================================

def select_images(image_files, requested_count):
    """
    Randomly select unique plate images while spreading selections across
    different CCPD subfolders and provinces where possible.
    """

    records = []
    seen_plates = set()

    for image_path in image_files:
        plate = get_plate_from_filename(image_path.name)
        if not plate or plate in seen_plates:
            continue

        seen_plates.add(plate)
        records.append((image_path, plate))

    if not records:
        return []

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(records)

    # First try to get different provinces/subfolders.
    selected = []
    used_provinces = set()
    remaining = []

    for image_path, plate in records:
        province = get_province_from_filename(image_path.name)

        if len(selected) < requested_count and province not in used_provinces:
            selected.append((image_path, plate))
            used_provinces.add(province)
        else:
            remaining.append((image_path, plate))

    for record in remaining:
        if len(selected) >= requested_count:
            break
        selected.append(record)

    return selected[:requested_count]


# ============================================================
# RECOGNIZE SELECTED PLATES
# ============================================================

def recognize_selected(model, selected):
    results = []

    for start in range(0, len(selected), BATCH_SIZE):
        batch = selected[start:start + BATCH_SIZE]

        tensors = []
        valid_records = []

        for image_path, actual_plate in batch:
            try:
                tensors.append(preprocess_image(image_path))
                valid_records.append((image_path, actual_plate))
            except Exception as error:
                print(f"\nWarning: Could not read {image_path.name}: {error}")

        if not tensors:
            continue

        try:
            batch_tensor = torch.stack(tensors).to(DEVICE)

            with torch.no_grad():
                output = model(batch_tensor)
                recognition = output["recognition"]

            predictions = decode_ctc_batch(recognition)

        except RuntimeError as error:
            print(f"\nBatch inference failed: {error}")
            print("Retrying this batch one image at a time...")
            predictions = []

            for tensor in tensors:
                try:
                    with torch.no_grad():
                        output = model(tensor.unsqueeze(0).to(DEVICE))
                        prediction = decode_ctc_single(output["recognition"])
                    predictions.append(prediction)
                except Exception:
                    predictions.append(("", 0.0))

        for (image_path, actual_plate), (model_prediction, confidence) in zip(
            valid_records, predictions
        ):
            model_prediction = clean_plate(model_prediction)
            color = detect_plate_color(image_path)
            category = classify_vehicle(color)
            province = get_province_from_filename(image_path.name)

            results.append({
                "plate": actual_plate,
                "model_prediction": model_prediction,
                "match": normalize_plate(model_prediction) == normalize_plate(actual_plate),
                "color": color,
                "category": category,
                "state": province,
                "confidence": confidence,
                "image": str(image_path),
            })

    return results[:len(selected)]


# ============================================================
# SAVE INDEX
# ============================================================

def save_index(results):
    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)

    with INDEX_FILE.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4)


# ============================================================
# SEARCH COMPLETE DATASET
# ============================================================

def search_dataset():
    print()
    print("=" * 60)
    print("                 PLATE SEARCH")
    print("=" * 60)

    query = input("Enter license plate number to search: ").strip()
    query = normalize_plate(query)

    if not query:
        print("No valid plate number entered.")
        return

    image_files = get_dataset_images()
    matches = []

    for image_path in image_files:
        actual_plate = get_plate_from_filename(image_path.name)

        if query in actual_plate:
            color = detect_plate_color(image_path)
            matches.append({
                "plate": actual_plate,
                "color": color,
                "category": classify_vehicle(color),
                "state": get_province_from_filename(image_path.name),
                "image": str(image_path),
            })

    print()

    if not matches:
        print("RESULT: Plate NOT FOUND in the complete CCPD dataset.")
        return

    print("RESULT: Plate FOUND in the complete CCPD dataset.")
    print("Matches found:", len(matches))
    print()

    for index, match in enumerate(matches[:20], start=1):
        print(f"Match {index}")
        print("License Plate Number:", match["plate"])
        print("Plate Color         :", match["color"])
        print("Vehicle Category    :", match["category"])
        print("State / Province    :", match["state"])
        print("Image               :", match["image"])
        print("-" * 60)

    if len(matches) > 20:
        print(f"Showing first 20 of {len(matches)} matches.")


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(results):
    print()
    print("=" * 60)
    print("                 ALPR SYSTEM RESULT")
    print("=" * 60)

    if not results:
        print("No results were generated.")
        return

    for index, result in enumerate(results, start=1):
        print()
        print(f"License Plate {index}")
        print("License Plate Number:", result["plate"])
        print("Model Prediction     :", result["model_prediction"] or "No prediction")
        print("Plate Color          :", result["color"])
        print("Vehicle Category     :", result["category"])
        print("State / Province     :", result["state"])
        print("Model Confidence     : {:.2f}%".format(result["confidence"] * 100))
        print("Prediction Match     :", "YES" if result["match"] else "NO")
        print("Image                :", result["image"])
        print("-" * 60)

    print()
    print("NOTE: License Plate Number is the exact CCPD dataset label extracted")
    print("      from the image filename. Model Prediction is the text produced")
    print("      by your trained ALPR model. Confidence belongs to the model")
    print("      prediction, not to the dataset label.")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 60)
    print("             LIGHT-EDGE ALPR SYSTEM")
    print("=" * 60)
    print("Dataset    :", DATASET_DIR)
    print("Checkpoint :", CHECKPOINT)
    print("Device     :", DEVICE)
    print()
    print("The License Plate Number shown in the result is the exact CCPD")
    print("dataset label. The Model Prediction shows what your current")
    print("trained ALPR model actually recognized.")

    while True:
        value = input("\nEnter number of license plates to detect: ").strip()
        try:
            requested_count = int(value)
            if requested_count > 0:
                break
        except ValueError:
            pass
        print("Please enter a positive whole number, such as 10, 60, or 100.")

    image_files = get_dataset_images()
    print("\nTotal dataset images found:", len(image_files))

    if requested_count > len(image_files):
        requested_count = len(image_files)
        print("Requested count was larger than the dataset; using", requested_count)

    selected = select_images(image_files, requested_count)

    if len(selected) < requested_count:
        print(
            f"Only {len(selected)} unique plate labels were available for selection."
        )

    model = load_model()

    print()
    print("Recognizing selected plates...")

    results = recognize_selected(model, selected)
    save_index(results)
    display_results(results)

    while True:
        choice = input("\nDo you want to search for a specific plate? (y/n): ").strip().lower()

        if choice == "y":
            search_dataset()
        elif choice == "n":
            print("\nProgram finished.")
            break
        else:
            print("Please enter y or n.")


if __name__ == "__main__":
    main()
