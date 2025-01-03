import os
import json
import numpy as np
from rotation import rotation_image
from text_detection import text_detection
from text_recognition import crop_text_by_bboxes
from key_info_extraction import key_info_extraction

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def process_image(image_path, json_file_path, crop_image_folder, final_json, result_json, threshold=0.10):
    image_name = os.path.basename(image_path)

    # Rotate the image
    rotated_image = rotation_image(image_path)

    # Initialize JSON data
    data = {}

    # Append the new image path
    data['image_path'] = [image_path]

    # Text detection
    bboxes = text_detection(rotated_image)
    if isinstance(bboxes, np.ndarray):
        bboxes = bboxes.tolist()
    data['bboxes'] = bboxes

    # Text recognition
    if not os.path.exists(crop_image_folder):
        os.makedirs(crop_image_folder)
    else:
        for file in os.listdir(crop_image_folder):
            os.remove(os.path.join(crop_image_folder, file))

    recognized_texts = crop_text_by_bboxes(image_name, rotated_image, bboxes, crop_image_folder)
    if isinstance(recognized_texts, list):
        recognized_texts = [item.tolist() if isinstance(item, np.ndarray) else item for item in recognized_texts]
    data['words'] = recognized_texts

    # Write the updated data to the JSON file
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

    # Key information extraction
    key_info_extraction(json_file_path, threshold, final_json)

    # Append results to the final result JSON
    if os.path.exists(result_json):
        with open(result_json, 'r') as json_file:
            try:
                existing_data = json.load(json_file)
            except json.JSONDecodeError:
                existing_data = []
    else:
        existing_data = []

    with open(final_json, 'r') as json_file:
        final_result = json.load(json_file)
        existing_data.append(final_result)

    with open(result_json, 'w', encoding='utf-8') as json_file:
        json.dump(existing_data, json_file, ensure_ascii=False, indent=4)

def main():
    input_json = '/Users/hungnguyen/Developer/repos/tensorflow/Final/MC_OCR/mc_ocr/data/test_data.json'
    crop_image_folder = '/Users/hungnguyen/Developer/repos/tensorflow/Final/MC_OCR/mc_ocr/main/crop_image_text'
    json_file_path = '/Users/hungnguyen/Developer/repos/tensorflow/Final/MC_OCR/mc_ocr/main/json_for_result.json'
    final_json = '/Users/hungnguyen/Developer/repos/tensorflow/Final/MC_OCR/mc_ocr/main/result_with_threshold.json'
    result_json = '/Users/hungnguyen/Developer/repos/tensorflow/Final/MC_OCR/mc_ocr/data/result.json'

    # Read the test JSON
    with open(input_json, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    # Extract all image paths
    image_paths = [entry['image_path'] for entry in json_data]

    i = 0
    # Process each image
    for image_path in image_paths:
        image_path = "/Users/hungnguyen/Developer/repos/tensorflow/Final/MC_OCR/mc_ocr/data/mc_ocr_train_filtered/" + image_path
        process_image(image_path, json_file_path, crop_image_folder, final_json, result_json)
        
        i += 1
        print(f"Processed image {i}")


if __name__ == '__main__':
    main()
