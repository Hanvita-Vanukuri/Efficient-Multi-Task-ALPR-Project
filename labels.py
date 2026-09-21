# ============================================================
# labels.py
# CCPD License Plate Character Mapping
# ============================================================

# Official CCPD character mapping used by your project
CCPD_CHARACTERS = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽",
    "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁",
    "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵",
    "云", "藏", "陕", "甘", "青", "宁", "新", "警",
    "学", "O"
]

# English letters
ALPHABET = list(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
)

# Digits
DIGITS = list(
    "0123456789"
)

# Complete character vocabulary
CHARACTERS = (
    CCPD_CHARACTERS
    + ALPHABET
    + DIGITS
)


# ============================================================
# CTC LABEL MAPPING
# ============================================================

# IMPORTANT:
# Index 0 is reserved for the CTC blank token.
char_to_idx = {
    character: index + 1
    for index, character in enumerate(CHARACTERS)
}

idx_to_char = {
    index: character
    for character, index in char_to_idx.items()
}


# ============================================================
# ENCODE LABEL
# ============================================================

def encode_label(label):
    """
    Convert a license plate string into integer labels.

    Example:
        "皖A33B52"
        -> [1, 34, ...]
    """

    encoded = []

    for character in label:

        if character not in char_to_idx:
            raise ValueError(
                f"Unknown character '{character}' "
                f"in license plate '{label}'."
            )

        encoded.append(
            char_to_idx[character]
        )

    return encoded


# ============================================================
# DECODE LABEL
# ============================================================

def decode_label(indices):
    """
    Convert integer labels back into a license plate string.

    CTC blank index 0 is ignored.
    """

    decoded = []

    for index in indices:

        # Ignore CTC blank
        if index == 0:
            continue

        if index in idx_to_char:
            decoded.append(
                idx_to_char[index]
            )

    return "".join(decoded)


# ============================================================
# INFORMATION
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("              CCPD LABEL INFORMATION")
    print("=" * 60)

    print(
        "CCPD characters       :",
        len(CCPD_CHARACTERS)
    )

    print(
        "English letters       :",
        len(ALPHABET)
    )

    print(
        "Digits                :",
        len(DIGITS)
    )

    print(
        "Total characters      :",
        len(CHARACTERS)
    )

    print(
        "CTC blank index       :",
        0
    )

    print(
        "Total model classes   :",
        len(CHARACTERS) + 1
    )

    print("=" * 60)

    # Test encoding/decoding
    test_plate = "皖A33B52"

    encoded = encode_label(
        test_plate
    )

    decoded = decode_label(
        encoded
    )

    print(
        "Test plate            :",
        test_plate
    )

    print(
        "Encoded               :",
        encoded
    )

    print(
        "Decoded               :",
        decoded
    )

    print("=" * 60)