import os
from PIL import Image
import pytesseract
import cv2
import re
import numpy as np
import json

def detect_orientation_with_tesseract(image_path):
    osd_result = pytesseract.image_to_osd(image_path, config='--psm 0 -c min_characters_to_try=5')
    print("Tesseract OSD Output:\n", osd_result)
    
    # Extract angle and confidence from OSD result
    angle = re.search(r'Orientation in degrees: \d+', osd_result).group().split(':')[-1].strip()
    confidence = re.search(r'Orientation confidence: \d+\.\d+', osd_result).group().split(':')[-1].strip()
    confidence = float(confidence)

    print(f"Detected Angle: {angle}, Confidence: {confidence}")
    return angle, confidence

def rotate_image_based_on_angle(image, angle, confidence):
    image_np = np.array(image)
    angle = int(angle)  # Convert angle to integer for comparison

    # If confidence is above a lower threshold (e.g., 0.5), or remove confidence check
    if angle == 90:
        rotated_image = cv2.rotate(image_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
        print("Image rotated 90 degrees counterclockwise.")
    elif angle == 180:
        rotated_image = cv2.rotate(image_np, cv2.ROTATE_180)
        print("Image rotated 180 degrees.")
    elif angle == 270:
        rotated_image = cv2.rotate(image_np, cv2.ROTATE_90_CLOCKWISE)
        print("Image rotated 90 degrees clockwise.")
    else:
        rotated_image = image_np
        print("Image is at the correct orientation (no rotation needed).")

    return rotated_image

def rotation_image(image_path):
    # Read the image using OpenCV
    image = cv2.imread(image_path)

    img_name = image_path.split('/')[-1]
    # cropped_image = crop_image(image)

    # Detect orientation using Tesseract OSD
    angle, confidence = detect_orientation_with_tesseract(image)

    # Rotate image based on detected angle and confidence
    rotated_image = rotate_image_based_on_angle(image, angle, confidence)

    return rotated_image


