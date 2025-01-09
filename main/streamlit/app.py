import os
import json
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from rotation import rotation_image
from text_detection import text_detection
from text_recognition import crop_text_by_bboxes
from key_info_extraction import key_info_extraction

def process_image(image_path):
    # Rotate the image
    rotated_image = rotation_image(image_path)

    # Text detection
    bboxes = text_detection(rotated_image)
    if isinstance(bboxes, np.ndarray):
        bboxes = bboxes.tolist()

    # Save the JSON data for image path and bounding boxes
    detection_json = '/Users/hungnguyen/Developer/repos/tensorflow/Final/MC_OCR/mc_ocr/main/streamlit/trash_folder/detection_json.json'
    # Clear the existing data by overwriting the file with an empty JSON object
    with open(detection_json, 'w') as json_file:
        json.dump({}, json_file)
    data = {}

    if 'image_path' not in data:
        data['image_path'] = []
    data['image_path'].append(image_path)
    data['bboxes'] = bboxes

    # Text recognition
    crop_image_folder = 'crop_image_text'
    if not os.path.exists(crop_image_folder):
        os.makedirs(crop_image_folder)
    else:
        files = os.listdir(crop_image_folder)
        for file in files:
            os.remove(os.path.join(crop_image_folder, file))

    recognized_texts = []
    recognized_texts = crop_text_by_bboxes(os.path.basename(image_path), rotated_image, bboxes, crop_image_folder)

    data['words'] = []

    if isinstance(recognized_texts, np.ndarray):
        recognized_texts = recognized_texts.tolist()
    elif isinstance(recognized_texts, list):
        recognized_texts = [item.tolist() if isinstance(item, np.ndarray) else item for item in recognized_texts]

    data['words'].extend(recognized_texts)

    with open(detection_json, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)

    # Key information extraction using LayoutLMv2
    final_json = '/Users/hungnguyen/Developer/repos/tensorflow/Final/MC_OCR/mc_ocr/main/streamlit/trash_folder/result_with_threshold.json'
    threshold = 0.10
    key_info_extraction(detection_json, threshold, final_json)

    with open(final_json, 'r') as json_file:
        final_result = json.load(json_file)

    return final_result, rotated_image

def main():
    st.title('OCR and Key Information Extraction')
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Process the image and display the result
        if st.button("Process Image"):
            with st.spinner('Processing image...'):
                image_path = f"./trash_folder/temp_{uploaded_file.name}"
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                final_result, processed_image = process_image(image_path)
                
                st.subheader('Extracted Tags and Words:')
                for word, tag in zip(final_result['words'], final_result['ner_tags']):
                  st.write(f"{tag} - {word}")

if __name__ == '__main__':
    main()
