from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms


# ============================================================
# CCPD DATASET PATH
# ============================================================

CCPD_DIR = Path.home() / "Desktop" / "CCPD-Dataset"

TRAIN_FILE = CCPD_DIR / "splits" / "ccpd_base_train.txt"
TEST_FILE = CCPD_DIR / "splits" / "ccpd_base_test.txt"


# ============================================================
# CHARACTER VOCABULARY
# ============================================================

# CCPD characters
CHARS = [
    "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽",
    "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁",
    "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵",
    "云", "藏", "陕", "甘", "青", "宁", "新",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z",
    "港", "澳"
]

# Remove duplicates while preserving order
CHARS = list(dict.fromkeys(CHARS))

CHAR_TO_IDX = {char: idx for idx, char in enumerate(CHARS)}
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}

# CTC blank class
BLANK_IDX = len(CHARS)

NUM_CLASSES = len(CHARS) + 1


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

IMAGE_WIDTH = 160
IMAGE_HEIGHT = 32

transform = transforms.Compose([
    transforms.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD SPLIT FILE
# ============================================================

def load_samples(split_file):
    """
    Load image paths and license plate labels
    from a CCPD split file.
    """

    split_file = Path(split_file)

    print("Reading split file:")
    print(split_file)
    print("Exists:", split_file.exists())

    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found:\n{split_file}"
        )

    samples = []

    with open(split_file, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 2:
                continue

            image_path = CCPD_DIR / parts[0]
            plate_number = parts[1]

            if image_path.exists():
                samples.append(
                    (image_path, plate_number)
                )

    return samples


# ============================================================
# ENCODE LICENSE PLATE
# ============================================================

def encode_plate(plate):

    labels = []

    for char in plate:

        if char in CHAR_TO_IDX:
            labels.append(CHAR_TO_IDX[char])

    return labels


# ============================================================
# DECODE LICENSE PLATE
# ============================================================

def decode_plate(indices):

    result = []

    for idx in indices:

        if idx == BLANK_IDX:
            continue

        if idx in IDX_TO_CHAR:
            result.append(IDX_TO_CHAR[idx])

    return "".join(result)


# ============================================================
# CCPD DATASET
# ============================================================

class CCPDDataset(Dataset):

    def __init__(
        self,
        split_file,
        transform=None
    ):

        self.transform = transform

        self.samples = load_samples(split_file)

        print(
            f"Loaded {len(self.samples)} samples"
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        image_path, plate_number = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = encode_plate(plate_number)

        labels = torch.tensor(
            labels,
            dtype=torch.long
        )

        return image, labels, plate_number


# ============================================================
# CTC COLLATE FUNCTION
# ============================================================

def ctc_collate_fn(batch):

    images = []
    labels = []
    plate_numbers = []

    for image, label, plate_number in batch:

        images.append(image)
        labels.append(label)
        plate_numbers.append(plate_number)

    images = torch.stack(images)

    label_lengths = torch.tensor(
        [len(label) for label in labels],
        dtype=torch.long
    )

    if len(labels) > 0:

        labels_concat = torch.cat(labels)

    else:

        labels_concat = torch.tensor(
            [],
            dtype=torch.long
        )

    return (
        images,
        labels_concat,
        label_lengths,
        plate_numbers
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CCPD DATASET TEST")
    print("=" * 60)

    print()
    print("Dataset:", CCPD_DIR)
    print("Train file:", TRAIN_FILE)
    print("Test file:", TEST_FILE)

    print()
    print("Train file exists:", TRAIN_FILE.exists())
    print("Test file exists:", TEST_FILE.exists())

    print()
    print("Characters:", len(CHARS))
    print("Number of classes:", NUM_CLASSES)

    # Load train dataset
    train_dataset = CCPDDataset(
        TRAIN_FILE,
        transform=transform
    )

    # Load test dataset
    test_dataset = CCPDDataset(
        TEST_FILE,
        transform=transform
    )

    print()
    print("=" * 60)
    print("DATASET LOADED SUCCESSFULLY")
    print("=" * 60)

    print()
    print("Training samples:", len(train_dataset))
    print("Testing samples:", len(test_dataset))

    # Test first sample
    image, labels, plate = train_dataset[0]

    print()
    print("First training sample:")
    print("Plate:", plate)
    print("Image tensor shape:", image.shape)
    print("Encoded labels:", labels.tolist())
    print(
        "Decoded:",
        decode_plate(labels.tolist())
    )

    print()
    print("Dataset test completed successfully.")