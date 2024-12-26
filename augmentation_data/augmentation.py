import os
import random
import cv2
import numpy as np
from shutil import copyfile
import ast
import re

# Define the augmentation function (you can expand this with more augmentations)
def augment_image(image, bboxes, background_image_path):
    # Example: Rotate the image and adjust bounding boxes accordingly
    angle = random.choice([90, 180, 270])  # You can rotate by these angles or pick others
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    
    # Rotate the bounding boxes (adjust for each box)
    rotated_bboxes = []
    for bbox in bboxes:
        pts = np.array(bbox).reshape((-1, 2))
        rotated_pts = cv2.transform(np.array([pts]), rotation_matrix)[0]
        rotated_bboxes.append(rotated_pts.flatten().tolist())
    
    return rotated_image, rotated_bboxes

# Load the bounding boxes from the file
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

# Process the directory
def process_directory(input_image_dir, bbox_dir, output_dir, background_image_path):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all bounding boxes from the file
    bboxes = load_bboxes(bbox_dir)

    for image_name, image_bboxes in bboxes:
        # Construct full path to the image
        image_path = os.path.join(input_image_dir, image_name)
        if not os.path.exists(image_path):
            print(f"Image {image_name} does not exist.")
            continue

        # Load the image
        image = cv2.imread(image_path)
        
        # Augment the image and bounding boxes
        augmented_image, augmented_bboxes = augment_image(image, image_bboxes, background_image_path)
        
        # Save the augmented image to the output directory
        augmented_image_name = f"aug_{image_name}"
        augmented_image_path = os.path.join(output_dir, augmented_image_name)
        cv2.imwrite(augmented_image_path, augmented_image)

        # Save the corresponding bounding boxes in a new file
        augmented_bbox_file = augmented_image_path.replace('.jpg', '.txt')
        with open(augmented_bbox_file, 'w') as f:
            for bbox in augmented_bboxes:
                f.write(f"{augmented_image_name}, {', '.join(map(str, bbox))}\n")
        
        print(f"Processed and augmented {image_name}")

# Set paths
input_image_dir = '../text_detector/rotation_image'  # Directory containing input images
bbox_dir = '../text_detector/boxes.txt'  # File containing bounding boxes
output_dir = '../augmentation_data/augmented_data'  # Directory to store augmented data
background_image_path = './background.jpg'  # Path to the background image (if needed for augmentation)

# Process the directory and augment data
process_directory(input_image_dir, bbox_dir, output_dir, background_image_path)