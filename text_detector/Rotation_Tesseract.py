# from PIL import Image
# import matplotlib.pyplot as plt
# import pytesseract
# import cv2
# import re
# import numpy as np

# def detect_orientation_with_tesseract(image_path):
#     osd_result = pytesseract.image_to_osd(image_path, config='--psm 0 -c min_characters_to_try=5')
#     print("Tesseract OSD Output:\n", osd_result)
    
#     # Extract angle and confidence from OSD result
#     angle = re.search(r'Orientation in degrees: \d+', osd_result).group().split(':')[-1].strip()
#     confidence = re.search(r'Orientation confidence: \d+\.\d+', osd_result).group().split(':')[-1].strip()
#     confidence = float(confidence)

#     print(f"Detected Angle: {angle}, Confidence: {confidence}")
#     return angle, confidence

# def crop_image(image):
#     height, width = image.shape[:2]
#     cropped_image = image[int(height * 0.1):int(height * 0.9), int(width * 0.1):int(width * 0.9)]  
#     return cropped_image

# def rotate_image_based_on_angle(image, angle, confidence):
#     image_np = np.array(image)
#     angle = int(angle)  # Convert angle to integer for comparison

#     # If confidence is above a lower threshold (e.g., 0.5), or remove confidence check
#     if angle == 90:
#         rotated_image = cv2.rotate(image_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
#         print("Image rotated 90 degrees counterclockwise.")
#     elif angle == 180:
#         rotated_image = cv2.rotate(image_np, cv2.ROTATE_180)
#         print("Image rotated 180 degrees.")
#     elif angle == 270:
#         rotated_image = cv2.rotate(image_np, cv2.ROTATE_90_CLOCKWISE)
#         print("Image rotated 90 degrees clockwise.")
#     else:
#         rotated_image = image_np
#         print("Image is at the correct orientation (no rotation needed).")

#     return rotated_image

# output_path = './rotation_image'
# image_path = '/Users/hungnguyen/Developer/repos/tensorflow/Final/180.jpg'
# image = cv2.imread(image_path)
# img_name = image_path.split('/')[-1]

# if image is None:
#     raise FileNotFoundError(f"Image not found at path: {image_path}")

# cropped_image = crop_image(image)

# # Detect orientation using Tesseract OSD
# angle, confidence = detect_orientation_with_tesseract(cropped_image)

# # Rotate image based on detected angle and confidence
# rotated_image = rotate_image_based_on_angle(cropped_image, angle, confidence)

# rotated_image_path = f'{output_path}/rotated_{img_name}'
# cv2.imwrite(rotated_image_path, rotated_image)
# print(f"Rotated image saved at: {rotated_image_path}")


import os
from PIL import Image
import pytesseract
import cv2
import re
import numpy as np
import matplotlib.pyplot as plt

def detect_orientation_with_tesseract(image_path):
    osd_result = pytesseract.image_to_osd(image_path, config='--psm 0 -c min_characters_to_try=5')
    print("Tesseract OSD Output:\n", osd_result)
    
    # Extract angle and confidence from OSD result
    angle = re.search(r'Orientation in degrees: \d+', osd_result).group().split(':')[-1].strip()
    confidence = re.search(r'Orientation confidence: \d+\.\d+', osd_result).group().split(':')[-1].strip()
    confidence = float(confidence)

    print(f"Detected Angle: {angle}, Confidence: {confidence}")
    return angle, confidence

def crop_image(image):
    height, width = image.shape[:2]
    cropped_image = image[int(height * 0.1):int(height * 0.9), int(width * 0.1):int(width * 0.9)]  
    return cropped_image

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

def visualize_image(image, title="Image"):
    """Display image using matplotlib."""
    plt.figure(figsize=(10,10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    plt.title(title)
    plt.axis('off')  # Hide axes
    plt.show()

image_path = '/Users/hungnguyen/Developer/repos/tensorflow/Final/MC_OCR/mc_ocr/data/mc_ocr_train/mcocr_public_145013aaprl.jpg'  

# Read the image using OpenCV
image = cv2.imread(image_path)

img_name = image_path.split('/')[-1]
cropped_image = crop_image(image)

# Detect orientation using Tesseract OSD
angle, confidence = detect_orientation_with_tesseract(cropped_image)

# Rotate image based on detected angle and confidence
rotated_image = rotate_image_based_on_angle(cropped_image, angle, confidence)

# Visualize the result
visualize_image(rotated_image, title="Rotated Image")
