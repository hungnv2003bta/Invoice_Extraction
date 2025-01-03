import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import re
import numpy as np
import json
from rotation import rotation_image
from text_detection import text_detection
from text_recognition import crop_text_by_bboxes
from key_info_extraction import key_info_extraction

def main():
    image_path = '/Users/hungnguyen/Developer/repos/tensorflow/Final/MC_OCR/mc_ocr/data/mc_ocr_train_filtered/mcocr_public_145014apodg.jpg'
    image_name = os.path.basename(image_path)
#---------------------------------------------------
    # ROTATE THE IMAGE
    rotated_image = rotation_image(image_path)
    # Append image path to JSON file
    json_file_path = '/Users/hungnguyen/Developer/repos/tensorflow/Final/MC_OCR/mc_ocr/main/json_for_result.json'

    data = {}

    # Read existing data from the JSON file if it exists
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as json_file:
            try:
                data = json.load(json_file)
            except json.JSONDecodeError:
                data = {}
    # if json exist, remove all data in it
    data.clear()
    # Append the new image path
    if 'image_path' not in data:
        data['image_path'] = []
    data['image_path'].append(image_path)
#---------------------------------------------------
    # Text detection
    bboxes = text_detection(rotated_image)

    # Convert bboxes to a list if it's a NumPy array
    if isinstance(bboxes, np.ndarray):
        bboxes = bboxes.tolist()

    data['bboxes'] = bboxes
#---------------------------------------------------
    # Text recognition
    crop_image_folder = '/Users/hungnguyen/Developer/repos/tensorflow/Final/MC_OCR/mc_ocr/main/crop_image_text'
    
    # if folder not exist, create it
    if not os.path.exists(crop_image_folder):
        os.makedirs(crop_image_folder)
    #if folder exist, remove all files in folder
    else:
        files = os.listdir(crop_image_folder)
        for file in files:
            os.remove(os.path.join(crop_image_folder, file))

    recognized_texts = crop_text_by_bboxes(image_name, rotated_image, bboxes, crop_image_folder)

    # Update recognized texts in JSON file
    if 'words' not in data:
        data['words'] = []

    # Convert recognized_texts elements to lists if needed
    if isinstance(recognized_texts, np.ndarray):
        recognized_texts = recognized_texts.tolist()
    elif isinstance(recognized_texts, list):
        recognized_texts = [item.tolist() if isinstance(item, np.ndarray) else item for item in recognized_texts]

    data['words'].extend(recognized_texts)
    # Write the updated data back to the JSON file
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
#---------------------------------------------------
    # KEY INFORMATION EXTRACTION USING LAYOUTLMv2   
    final_json = '/Users/hungnguyen/Developer/repos/tensorflow/Final/MC_OCR/mc_ocr/main/result_with_threshold.json'
    threshold = 0.10
    key_info_extraction(json_file_path, threshold, final_json)

    # Read the final result from the JSON file
    with open(final_json, 'r') as json_file:
        final_result = json.load(json_file)
        print(final_result)
        




if __name__ == '__main__':
    main()