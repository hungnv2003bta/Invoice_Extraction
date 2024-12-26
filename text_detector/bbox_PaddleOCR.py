# GET BBX COORDINATES OF TEXT IN IMAGE

# pip install paddleocr
# pip install paddlepaddle

import os
from paddleocr import PaddleOCR
import cv2
# Plot the image and bounding boxes
import matplotlib.pyplot as plt

def process_image(image_path, lang='vi'):
    # Initialize the OCR model
    ocr = PaddleOCR(use_angle_cls=True, lang=lang)
    
    # Perform OCR on the image
    result = ocr.ocr(image_path, cls=True)
    
    # Read the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Plot the image
    plt.imshow(image)
    
    # Plot the bounding boxes
    for line in result[0]:
        bbox = line[0]
        bbox = [(int(coord[0]), int(coord[1])) for coord in bbox]
        bbox.append(bbox[0])  # Close the polygon
        
        xs, ys = zip(*bbox)
        plt.plot(xs, ys, 'r', linewidth=2)
    
    plt.show()

# Example usage:
input_image_path = '/Users/hungnguyen/Developer/repos/tensorflow/Final/MC_OCR/mc_ocr/data/mc_ocr_train/mcocr_public_145013aaprl.jpg'  
process_image(input_image_path)
