import cv2
import numpy as np


# ============================================================
# 1. DETECT LICENSE PLATE COLOR
# ============================================================

def detect_plate_color(plate_crop):
    """
    Detect the dominant color of the license plate.

    Returns:
        plate_color
        vehicle_type
    """

    if plate_crop is None or plate_crop.size == 0:
        return "Unknown", "Unknown"

    # Resize for stable color analysis
    plate_crop = cv2.resize(
        plate_crop,
        (300, 100)
    )

    # Convert BGR -> HSV
    hsv = cv2.cvtColor(
        plate_crop,
        cv2.COLOR_BGR2HSV
    )

    # --------------------------------------------------------
    # Yellow
    # --------------------------------------------------------

    yellow_lower = np.array([15, 70, 70])
    yellow_upper = np.array([40, 255, 255])

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
    # White
    # --------------------------------------------------------

    white_lower = np.array([0, 0, 130])
    white_upper = np.array([180, 70, 255])

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
    # Blue
    # --------------------------------------------------------

    blue_lower = np.array([90, 70, 50])
    blue_upper = np.array([140, 255, 255])

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
    # Green
    # --------------------------------------------------------

    green_lower = np.array([35, 50, 40])
    green_upper = np.array([90, 255, 255])

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
    # Compare color ratios
    # --------------------------------------------------------

    colors = {
        "Yellow": yellow_ratio,
        "White": white_ratio,
        "Blue": blue_ratio,
        "Green": green_ratio
    }

    detected_color = max(
        colors,
        key=colors.get
    )

    confidence = colors[detected_color]

    # --------------------------------------------------------
    # If color detection is uncertain
    # --------------------------------------------------------

    if confidence < 0.08:
        return "Unknown", "Unknown"

    # --------------------------------------------------------
    # Vehicle classification
    # --------------------------------------------------------

    if detected_color == "Yellow":

        vehicle_type = "Taxi / Commercial"

    elif detected_color == "White":

        vehicle_type = "Personal / Private"

    else:

        vehicle_type = "Other / Commercial"

    return detected_color, vehicle_type


# ============================================================
# 2. DRAW INFORMATION ON IMAGE
# ============================================================

def draw_alpr_information(
    image,
    bbox,
    plate_text
):

    x1, y1, x2, y2 = map(
        int,
        bbox
    )

    # --------------------------------------------------------
    # Keep coordinates inside image
    # --------------------------------------------------------

    height, width = image.shape[:2]

    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))

    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

    if x2 <= x1 or y2 <= y1:

        return image

    # --------------------------------------------------------
    # Crop license plate
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

    text_lines = [
        f"Plate: {plate_text}",
        f"Color: {plate_color}",
        f"Vehicle: {vehicle_type}"
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.65
    thickness = 2

    # --------------------------------------------------------
    # Calculate information box size
    # --------------------------------------------------------

    text_sizes = []

    for text in text_lines:

        size, _ = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness
        )

        text_sizes.append(size)

    box_width = max(
        size[0] for size in text_sizes
    ) + 20

    line_height = 25

    box_height = (
        len(text_lines) * line_height
    ) + 15

    # --------------------------------------------------------
    # Put information box above plate
    # --------------------------------------------------------

    box_x1 = x1

    box_y1 = y1 - box_height

    # If there isn't enough space above plate,
    # put box below plate
    if box_y1 < 0:

        box_y1 = y2 + 5

    box_x2 = box_x1 + box_width

    box_y2 = box_y1 + box_height

    # Keep box inside image
    if box_x2 > width:

        box_x2 = width

    if box_y2 > height:

        box_y2 = height

    # --------------------------------------------------------
    # Draw black background
    # --------------------------------------------------------

    cv2.rectangle(
        image,
        (box_x1, box_y1),
        (box_x2, box_y2),
        (0, 0, 0),
        -1
    )

    # --------------------------------------------------------
    # Draw text
    # --------------------------------------------------------

    text_y = box_y1 + 23

    for text in text_lines:

        cv2.putText(
            image,
            text,
            (box_x1 + 8, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

        text_y += line_height

    return image


# ============================================================
# 3. COMPLETE EXTENSION FUNCTION
# ============================================================

def add_vehicle_information(
    image,
    bbox,
    plate_text
):

    """
    Add:
        License plate bounding box
        License plate text
        Plate color
        Vehicle type

    to the original image.
    """

    result = draw_alpr_information(
        image,
        bbox,
        plate_text
    )

    return result