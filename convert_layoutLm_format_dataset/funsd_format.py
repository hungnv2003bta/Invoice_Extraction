import ast
import os
import pandas as pd
import json

# Function to convert CSV data to the required LayoutLM format
def convert_to_layoutlm_format(csv_file):
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Prepare a list to hold the converted data
    layoutlm_data = []
    
    for index, row in df.iterrows():
        img_id = row['img_id']
        anno_polygons = ast.literal_eval(row['anno_polygons'])  # Convert string to list of dictionaries
        anno_texts = row['anno_texts'].split('|||')  # Split the text by '|||'
        
        # Get the image path, assuming the images are stored in a folder 'images/'
        image_path = f"images/{img_id}"
        
        # Prepare words, bboxes, and ner_tags lists
        words = []
        bboxes = []
        ner_tags = []
        
        # Iterate through each annotation to extract the data
        for i, anno in enumerate(anno_polygons):
            text = anno_texts[i] if i < len(anno_texts) else ""  # Avoid out of range
            
            # Add the text to words list (ensure no duplicates)
            if text not in words:
                words.append(text)
            
            # Convert segmentation points to bounding boxes (x1, y1, x2, y2)
            bbox = anno['bbox']
            bboxes.append([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])  # [x1, y1, x2, y2]
            
            # Add the category_id (as ner_tag)
            ner_tags.append(anno['category_id'])  # Use category_id as ner_tag
        
        # Create the record for LayoutLM format
        layoutlm_record = {
            'id': img_id.split('.')[0],  # Extract the ID without extension
            'words': words,
            'bboxes': bboxes,
            'ner_tags': ner_tags,
            'image_path': image_path
        }
        
        # Append to the list of LayoutLM data
        layoutlm_data.append(layoutlm_record)
    
    return layoutlm_data

# Function to save the data to a JSON file in FUNSD format
def save_to_funsd_format(csv_file, output_file):
    layoutlm_data = convert_to_layoutlm_format(csv_file)
    
    # Save the data as a JSON file
    with open(output_file, 'w') as f:
        json.dump(layoutlm_data, f, ensure_ascii=False, indent=4)

# Example usage
csv_file = '/Users/hungnguyen/Developer/repos/tensorflow/Final/MC_OCR/mc_ocr/data/mcocr_train_df_filtered.csv'  # Replace with your actual CSV file path
output_file = '/Users/hungnguyen/Developer/repos/tensorflow/Final/MC_OCR/mc_ocr/data/funsd_format.json'  # Output JSON file path
save_to_funsd_format(csv_file, output_file)

print(f"Data has been saved to {output_file}")
