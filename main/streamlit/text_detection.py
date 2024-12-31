# pip install paddleocr
# pip install paddlepaddle
# pip install pytesseract

# TEXT DETECTION USING PADDLE OCR
from paddleocr import PaddleOCR
import os
from PIL import Image, ImageDraw
import cv2

def text_detection(rotated_image, lang='vi'):
    # Initialize the OCR model
    ocr = PaddleOCR(use_angle_cls=True, lang=lang)
    
    # Perform OCR on the image
    result = ocr.ocr(rotated_image, cls=True)
    
    # List to store bounding boxes
    bboxes = []
    
    # Loop through OCR results and extract bounding boxes
    for line in result[0]:
        bbox = line[0]
        bbox = [(int(coord[0]), int(coord[1])) for coord in bbox]
        bbox.append(bbox[0])  # Close the polygon by repeating the first point
        bboxes.append(bbox)  # Store the bounding box

    # Convert the rotated image to RGB (OpenCV uses BGR, PIL uses RGB)
    return bboxes
