import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm

from model import ALPRModel


# ============================================================
# CONFIGURATION
# ============================================================

CCPD_DIR = Path.home() / "Desktop" / "CCPD-Dataset"

TEST_FILE = CCPD_DIR / "splits" / "ccpd_base_test.txt"

MODEL_FILE = Path("results/alpr_final.pth")

IMAGE_SIZE = (32, 160)

BATCH_SIZE = 64

NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple GPU (MPS)")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")


# ============================================================
# LOAD TEST DATA
# ============================================================

def read_test_split(file_path):

    samples = []

    with open(file_path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            image_path = CCPD_DIR / parts[0]

            plate = parts[1]

            samples.append(
                (image_path, plate)
            )

    return samples


print("Reading test dataset...")

samples = read_test_split(TEST_FILE)

print("Test samples:", len(samples))


# ============================================================
# LOAD CHARACTER MAP
# ============================================================

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE
)

characters = checkpoint["characters"]

char_to_idx = {
    char: idx
    for idx, char in enumerate(characters)
}

idx_to_char = {
    idx: char
    for idx, char in enumerate(characters)
}

NUM_CLASSES = len(characters)

print("Number of characters:", NUM_CLASSES)


# ============================================================
# DATASET
# ============================================================

transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


class CCPDTestDataset(Dataset):

    def __init__(self, samples):

        self.samples = samples

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        image_path, plate = self.samples[index]

        image = Image.open(
            image_path
        ).convert("RGB")

        image = transform(image)

        label = torch.tensor(
            [char_to_idx[c] for c in plate],
            dtype=torch.long
        )

        return image, label, plate


# ============================================================
# DATALOADER
# ============================================================

dataset = CCPDTestDataset(samples)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False
)


# ============================================================
# MODEL
# ============================================================

model = ALPRModel(NUM_CLASSES)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)

model.eval()

print("\nModel loaded successfully.")


# ============================================================
# ACCURACY VARIABLES
# ============================================================

total_characters = 0

correct_characters = 0

total_plates = 0

correct_full_plates = 0


# ============================================================
# TESTING
# ============================================================

print("\nStarting testing...\n")


with torch.no_grad():

    progress = tqdm(
        loader,
        desc="Testing"
    )

    for images, labels, plates in progress:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        # Model prediction
        outputs = model(images)

        # Get highest probability character
        predictions = torch.argmax(
            outputs,
            dim=2
        )

        # ----------------------------------------------------
        # CHARACTER ACCURACY
        # ----------------------------------------------------

        correct_characters += (
            predictions == labels
        ).sum().item()

        total_characters += labels.numel()


        # ----------------------------------------------------
        # FULL PLATE ACCURACY
        # ----------------------------------------------------

        for i in range(len(plates)):

            predicted_indices = predictions[i]

            predicted_plate = "".join(
                idx_to_char[
                    idx.item()
                ]
                for idx in predicted_indices
            )

            ground_truth_plate = plates[i]

            total_plates += 1

            if predicted_plate == ground_truth_plate:

                correct_full_plates += 1


# ============================================================
# CALCULATE ACCURACY
# ============================================================

character_accuracy = (
    correct_characters /
    total_characters
) * 100


full_plate_accuracy = (
    correct_full_plates /
    total_plates
) * 100


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("                 ALPR TEST RESULTS")
print("=" * 60)

print(
    f"Total test plates       : {total_plates}"
)

print(
    f"Correct full plates     : {correct_full_plates}"
)

print(
    f"Total characters        : {total_characters}"
)

print(
    f"Correct characters      : {correct_characters}"
)

print(
    f"Character Accuracy      : {character_accuracy:.2f}%"
)

print(
    f"Full-Plate Accuracy     : {full_plate_accuracy:.2f}%"
)

print("=" * 60)


# ============================================================
# SAVE RESULTS
# ============================================================

with open(
    "results/test_results.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("ALPR TEST RESULTS\n")
    f.write("=" * 50 + "\n")

    f.write(
        f"Total test plates: {total_plates}\n"
    )

    f.write(
        f"Correct full plates: {correct_full_plates}\n"
    )

    f.write(
        f"Total characters: {total_characters}\n"
    )

    f.write(
        f"Correct characters: {correct_characters}\n"
    )

    f.write(
        f"Character Accuracy: "
        f"{character_accuracy:.2f}%\n"
    )

    f.write(
        f"Full-Plate Accuracy: "
        f"{full_plate_accuracy:.2f}%\n"
    )

print(
    "\nResults saved to:"
)

print(
    "results/test_results.txt"
)