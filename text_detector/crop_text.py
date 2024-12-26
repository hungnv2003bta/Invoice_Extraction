# CROP TEXTLINE based on the bounding box of the textline

import cv2
import numpy as np
import ast
import re

# Function to load bounding boxes from a text file
def load_bboxes(bbox_file):
    bboxes = []
    bbox_pattern = re.compile(r'\[([^\]]+)\]')  # Regex to match anything inside square brackets
    with open(bbox_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split(', ', 1)  # Split only on the first comma to get the image name
            image_name = parts[0]  # The image name is the first part
            bbox_strs = bbox_pattern.findall(parts[1])  # Find all bounding box lists
            
            # Parse the bounding box strings into actual lists of floats
            bbox = [list(map(float, bbox.split(', '))) for bbox in bbox_strs]
            bboxes.append((image_name, bbox))
    
    return bboxes

# Function to crop image based on bounding boxes
def crop_image(image, bbox):
    # Bounding box is in the form of [x1, y1, x2, y2, x3, y3, x4, y4]
    # We can convert these points into a bounding box that can be used for cropping
    # For simplicity, we will use the top-left and bottom-right coordinates.
    x_min = min(bbox[0], bbox[2], bbox[4], bbox[6])
    y_min = min(bbox[1], bbox[3], bbox[5], bbox[7])
    x_max = max(bbox[0], bbox[2], bbox[4], bbox[6])
    y_max = max(bbox[1], bbox[3], bbox[5], bbox[7])
    
    # Crop the image using the bounding box coordinates
    cropped_image = image[int(y_min):int(y_max), int(x_min):int(x_max)]
    return cropped_image

# Function to process images in a directory
def process_images(input_image_dir, bbox_dir, output_dir):
    # Load bounding boxes
    bboxes = load_bboxes(bbox_dir)

    for image_name, bbox_list in bboxes:
        image_path = f"{input_image_dir}/{image_name}"
        image = cv2.imread(image_path)  # Read the image

        if image is None:
            print(f"Error loading image: {image_path}")
            continue
        
        for i, bbox in enumerate(bbox_list):
            # Crop each text region
            cropped_image = crop_image(image, bbox)

            # Save the cropped image (you can save with unique names if necessary)
            output_image_path = f"{output_dir}/{image_name}_cropped_{i}.jpg"
            cv2.imwrite(output_image_path, cropped_image)
            print(f"Saved cropped image: {output_image_path}")

# Example usage:
input_image_dir = 'text_detector/rotation_image' 
bbox_dir = 'text_detector/boxes.txt'
output_dir = 'text_detector/crop_image_text' 

process_images(input_image_dir, bbox_dir, output_dir)
