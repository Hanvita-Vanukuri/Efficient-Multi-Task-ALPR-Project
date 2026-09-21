import cv2
import numpy as np


# ============================================================
# PLATE COLOR DETECTION
# ============================================================

def detect_plate_color(plate_image):
    """
    Detect the dominant license plate color.

    Returns:
        color_name
        vehicle_type
    """

    if plate_image is None or plate_image.size == 0:
        return "Unknown", "Unknown"

    # Resize for stable color analysis
    plate = cv2.resize(
        plate_image,
        (200, 80)
    )

    hsv = cv2.cvtColor(
        plate,
        cv2.COLOR_BGR2HSV
    )

    # --------------------------------------------------------
    # Yellow plate
    # --------------------------------------------------------

    yellow_lower = np.array(
        [15, 70, 70]
    )

    yellow_upper = np.array(
        [40, 255, 255]
    )

    yellow_mask = cv2.inRange(
        hsv,
        yellow_lower,
        yellow_upper
    )

    yellow_ratio = (
        np.count_nonzero(yellow_mask)
        / yellow_mask.size
    )

    # --------------------------------------------------------
    # Blue plate
    # --------------------------------------------------------

    blue_lower = np.array(
        [90, 70, 50]
    )

    blue_upper = np.array(
        [140, 255, 255]
    )

    blue_mask = cv2.inRange(
        hsv,
        blue_lower,
        blue_upper
    )

    blue_ratio = (
        np.count_nonzero(blue_mask)
        / blue_mask.size
    )

    # --------------------------------------------------------
    # Green plate
    # --------------------------------------------------------

    green_lower = np.array(
        [35, 50, 40]
    )

    green_upper = np.array(
        [90, 255, 255]
    )

    green_mask = cv2.inRange(
        hsv,
        green_lower,
        green_upper
    )

    green_ratio = (
        np.count_nonzero(green_mask)
        / green_mask.size
    )

    # --------------------------------------------------------
    # White plate
    # --------------------------------------------------------

    white_lower = np.array(
        [0, 0, 130]
    )

    white_upper = np.array(
        [180, 70, 255]
    )

    white_mask = cv2.inRange(
        hsv,
        white_lower,
        white_upper
    )

    white_ratio = (
        np.count_nonzero(white_mask)
        / white_mask.size
    )

    # --------------------------------------------------------
    # Select dominant color
    # --------------------------------------------------------

    ratios = {
        "Yellow": yellow_ratio,
        "Blue": blue_ratio,
        "Green": green_ratio,
        "White": white_ratio
    }

    color = max(
        ratios,
        key=ratios.get
    )

    confidence = ratios[color]

    # Avoid making a decision when color evidence is weak
    if confidence < 0.08:
        return "Unknown", "Unknown"

    # --------------------------------------------------------
    # Vehicle classification
    # --------------------------------------------------------

    if color == "Yellow":
        vehicle_type = "Taxi / Commercial"

    elif color == "White":
        vehicle_type = "Personal / Private"

    else:
        vehicle_type = "Other / Commercial"

    return color, vehicle_type


# ============================================================
# DRAW ALPR RESULT
# ============================================================

def draw_alpr_result(
    image,
    bbox,
    plate_text,
    plate_color,
    vehicle_type
):
    """
    Draw bounding box and ALPR information on image.

    bbox format:
        (x1, y1, x2, y2)
    """

    x1, y1, x2, y2 = map(
        int,
        bbox
    )

    # --------------------------------------------------------
    # Draw bounding box
    # --------------------------------------------------------

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    # --------------------------------------------------------
    # Information text
    # --------------------------------------------------------

    text1 = f"Plate: {plate_text}"
    text2 = f"Color: {plate_color}"
    text3 = f"Vehicle: {vehicle_type}"

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.65
    thickness = 2

    # Calculate text area
    texts = [
        text1,
        text2,
        text3
    ]

    widths = []

    heights = []

    for text in texts:

        (w, h), _ = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness
        )

        widths.append(w)
        heights.append(h)

    box_width = max(widths) + 20

    box_height = sum(heights) + 35

    # Put information above the plate
    label_x = x1

    label_y = max(
        y1 - box_height,
        0
    )

    # Background
    cv2.rectangle(
        image,
        (label_x, label_y),
        (
            label_x + box_width,
            label_y + box_height
        ),
        (0, 0, 0),
        -1
    )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    current_y = label_y + 22

    for text, height in zip(
        texts,
        heights
    ):

        cv2.putText(
            image,
            text,
            (
                label_x + 10,
                current_y
            ),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

        current_y += height + 4

    return image


# ============================================================
# COMPLETE ALPR VISUALIZATION
# ============================================================

def process_alpr_result(
    image,
    bbox,
    plate_text
):
    """
    Complete pipeline:

    1. Crop detected plate
    2. Detect plate color
    3. Classify vehicle
    4. Draw everything on original image
    """

    x1, y1, x2, y2 = map(
        int,
        bbox
    )

    height, width = image.shape[:2]

    # Keep coordinates inside image
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))

    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

    if x2 <= x1 or y2 <= y1:

        return image, "Unknown", "Unknown"

    # --------------------------------------------------------
    # Crop plate
    # --------------------------------------------------------

    plate_crop = image[
        y1:y2,
        x1:x2
    ]

    # --------------------------------------------------------
    # Detect color
    # --------------------------------------------------------

    plate_color, vehicle_type = detect_plate_color(
        plate_crop
    )

    # --------------------------------------------------------
    # Draw result
    # --------------------------------------------------------

    image = draw_alpr_result(
        image,
        bbox,
        plate_text,
        plate_color,
        vehicle_type
    )

    return (
        image,
        plate_color,
        vehicle_type
    )