from transformers import LayoutLMForTokenClassification, LayoutLMv2Processor
import os
import json
import cv2
from PIL import Image
import torch.nn.functional as F

# Load LayoutLM model and processor from Huggingface Hub
model = LayoutLMForTokenClassification.from_pretrained("HungNguyen142/layoutlm-mcocr")
processor = LayoutLMv2Processor.from_pretrained("HungNguyen142/layoutlm-mcocr")

# Normalize box function for LayoutLM
def unnormalize_box(bbox, width, height):
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]

# Color mapping for different labels
label2color = {
    "SELLER": "blue",
    "ADDRESS": "red",
    "TIMESTAMP": "green",
    "TOTAL_COST": "pink"
}

# Run inference on cropped image
def run_inference_on_cropped_image(cropped_image, threshold, model=model, processor=processor):
    # Convert the cropped image to RGB format
    if cropped_image is not None and cropped_image.size > 0:
        image = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)).convert("RGB")
    else:
        return cropped_image

    # Create model input
    encoding = processor(images=image, return_tensors="pt")
    del encoding["image"]

    # Run inference
    outputs = model(**encoding)

    # Apply softmax to logits to get probabilities
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1)

    # Get the labels based on the highest probability
    predictions = probabilities.argmax(-1).squeeze().tolist()
    labels = [model.config.id2label[prediction] for prediction in predictions]

    # Get probabilities for the predicted labels
    predicted_probabilities = probabilities.max(-1).values.squeeze().tolist()

    # Get the highest predicted probability
    highest_probability = max(predicted_probabilities)
    if highest_probability < float(threshold):
        return None
    else : 
        # Get the label with the highest probability
        highest_label = labels[predicted_probabilities.index(highest_probability)]
        filtered_labels = [highest_label]
        return filtered_labels, highest_probability
    
# Function to find valid total cost: a text and a number

def find_total_cost(valid_words_total_cost, prob_total_cost, valid_bboxes_total_cost, deviation_threshold=50):
    text_candidates = []
    number_candidates = []

    # Separate text and number candidates
    for word, prob, bbox in zip(valid_words_total_cost, prob_total_cost, valid_bboxes_total_cost):
        if any(char.isdigit() for char in word) and ',' in word:  # Number candidates with a comma
            number_candidates.append((word, prob, bbox))
        elif all(not char.isdigit() for char in word):  # Text candidates without digits
            text_candidates.append((word, prob, bbox))

    total_cost_text = None
    total_cost_number = None
    best_match_score = float('inf')
    bbox_text = None
    bbox_number = None

    # Check alignment for Text and Number
    for text, text_prob, text_bbox in text_candidates:
        text_y_center = (text_bbox[0][1] + text_bbox[2][1]) / 2  # Center Y of Text bounding box

        for number, number_prob, number_bbox in number_candidates:
            number_y_center = (number_bbox[0][1] + number_bbox[2][1]) / 2  # Center Y of Number bounding box

            # Compute vertical and horizontal proximity
            y_distance = abs(text_y_center - number_y_center)
            x_distance = abs((text_bbox[0][0] + text_bbox[2][0]) / 2 - (number_bbox[0][0] + number_bbox[2][0]) / 2)

            # Match if proximity satisfies the threshold
            if y_distance <= deviation_threshold or x_distance <= deviation_threshold:
                # Match score based on distance and probabilities
                match_score = y_distance + x_distance - (text_prob + number_prob)
                if match_score < best_match_score:
                    best_match_score = match_score
                    total_cost_text = text
                    total_cost_number = number
                    bbox_text = text_bbox
                    bbox_number = number_bbox

    return total_cost_text, total_cost_number, bbox_text, bbox_number

# Function to process bounding boxes and filter based on NER predictions
def process_json_and_run_inference(json_file_path, threshold, final_json):
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Load the image (you can handle rotated images here if needed)
    image_path = data['image_path'][0]

    # Open the final result JSON file and write the image path into it
    with open(final_json, 'w') as final_result_file:
        json.dump({"image_path": image_path}, final_result_file, indent=4)
    image = cv2.imread(image_path)

    rotated_image = image  

    # Initialize to store bbox, words, and NER tags for ADDRESS
    valid_bboxes_address = []
    valid_words_address = []
    ner_tags_address = []
    prob_address = []
    # Initialize to store bbox, words, and NER tags for SELLER
    valid_bboxes_seller = []
    valid_words_seller = []
    ner_tags_seller = []
    prob_seller = []
    # Initialize to store bbox, words, and NER tags for TOTAL_COST
    valid_bboxes_total_cost = []
    valid_words_total_cost = []
    ner_tags_total_cost = []
    prob_total_cost = []
    # Initialize to store bbox, words, and NER tags for TIMESTAMP
    valid_bboxes_timestamp = []
    valid_words_timestamp = []
    ner_tags_timestamp = []
    prob_timestamp = []

    label_counts = {
        "ADDRESS": 0,
        "SELLER": 0,
        "TOTAL_COST": 0,
        "TIMESTAMP": 0
    }

    # Iterate through each bounding box and process the corresponding crop
    for i, (bbox, word) in enumerate(zip(data['bboxes'], data['words'])):
        # Extract the coordinates from the bounding box
        x_min = min([point[0] for point in bbox])
        y_min = min([point[1] for point in bbox])
        x_max = max([point[0] for point in bbox])
        y_max = max([point[1] for point in bbox])
        
        # Crop the image using the bounding box coordinates
        cropped_image = rotated_image[y_min:y_max, x_min:x_max]
        
        # Perform inference on the cropped image
        inference_result = run_inference_on_cropped_image(cropped_image, threshold)

        # Proceed only if inference_result is not None
        if inference_result:
            recognized_labels, prob = inference_result

            if recognized_labels:
                for label in recognized_labels:
                    if label == "ADDRESS":
                        valid_bboxes_address.append(bbox)
                        valid_words_address.append(word)
                        ner_tags_address.append(label)
                        prob_address.append(prob)
                        label_counts["ADDRESS"] += 1
                    elif label == "SELLER":
                        valid_bboxes_seller.append(bbox)
                        valid_words_seller.append(word)
                        ner_tags_seller.append(label)
                        prob_seller.append(prob)
                        label_counts["SELLER"] += 1
                    elif label == "TOTAL_COST":
                        valid_bboxes_total_cost.append(bbox)
                        valid_words_total_cost.append(word)
                        ner_tags_total_cost.append(label)
                        prob_total_cost.append(prob)
                        label_counts["TOTAL_COST"] += 1
                    elif label == "TIMESTAMP":
                        valid_bboxes_timestamp.append(bbox)
                        valid_words_timestamp.append(word)
                        ner_tags_timestamp.append(label)
                        prob_timestamp.append(prob)
                        label_counts["TIMESTAMP"] += 1

    # with ADDRESS label, sort bbox, word by prob 
    valid_bboxes_address = [bbox for _, bbox in sorted(zip(prob_address, valid_bboxes_address), reverse=True)]
    valid_words_address = [word for _, word in sorted(zip(prob_address, valid_words_address), reverse=True)]
    # with SELLER label, sort bbox, word by prob
    valid_bboxes_seller = [bbox for _, bbox in sorted(zip(prob_seller, valid_bboxes_seller), reverse=True)]
    valid_words_seller = [word for _, word in sorted(zip(prob_seller, valid_words_seller), reverse=True)]
    # with TOTAL_COST label, sort bbox, word by prob
    valid_bboxes_total_cost = [bbox for _, bbox in sorted(zip(prob_total_cost, valid_bboxes_total_cost), reverse=True)]
    valid_words_total_cost = [word for _, word in sorted(zip(prob_total_cost, valid_words_total_cost), reverse=True)]
    # with TIMESTAMP label, sort bbox, word by prob
    valid_bboxes_timestamp = [bbox for _, bbox in sorted(zip(prob_timestamp, valid_bboxes_timestamp), reverse=True)]
    valid_words_timestamp = [word for _, word in sorted(zip(prob_timestamp, valid_words_timestamp), reverse=True)]

    result_bboxes = []
    result_words = []
    result_ner_tags = []

    seller_word = valid_words_seller[0] if valid_words_seller else None
    seller_bbox = valid_bboxes_seller[0] if valid_bboxes_seller else None
    result_words.append(seller_word)
    result_bboxes.append(seller_bbox)
    result_ner_tags.append("SELLER")

    time_stamp_word = valid_words_timestamp[0] if valid_words_timestamp else None
    time_stamp_bbox = valid_bboxes_timestamp[0] if valid_bboxes_timestamp else None
    result_words.append(time_stamp_word)
    result_bboxes.append(time_stamp_bbox)
    result_ner_tags.append("TIMESTAMP")

    total_cost_text, total_cost_number, bbox_text, bbox_number = find_total_cost(valid_words_total_cost, prob_total_cost, valid_bboxes_total_cost)
    if total_cost_text is not None and total_cost_number is not None:
        result_words.append(total_cost_text)
        result_bboxes.append(bbox_text)
        result_ner_tags.append("TOTAL_COST")
        result_words.append(total_cost_number)
        result_bboxes.append(bbox_number)
        result_ner_tags.append("TOTAL_COST")
    else:
        # Append the highest probability TOTAL_COST candidate
        if valid_words_total_cost:
            result_words.append(valid_words_total_cost[0])
            result_bboxes.append(valid_bboxes_total_cost[0])
            result_ner_tags.append("TOTAL_COST")
        else :
            result_words.append(None)
            result_bboxes.append(None)
            result_ner_tags.append("TOTAL_COST")

    # Check if address prob > 0.97, append all
    for word, prob, bbox in zip(valid_words_address, prob_address, valid_bboxes_address):
        if prob > 0.97:
            result_words.append(word)
            result_bboxes.append(bbox)
            result_ner_tags.append("ADDRESS")


    # Write valid bounding boxes, words, and NER tags into the final JSON file
    with open(final_json, 'r+', encoding='utf-8') as final_result_file:
        # Load existing data from the JSON file
        final_data = json.load(final_result_file)
        
        # Keep only the image_path and add new data
        final_data = {
            "image_path": final_data["image_path"],  # Keep the image_path
            "bboxes": result_bboxes,  # Add new bounding boxes
            "words": result_words,  # Add new words
            "ner_tags": result_ner_tags  # Add new NER tags
        }
        
        # Move the file pointer to the beginning of the file before writing
        final_result_file.seek(0)
        
        # Write the updated data to the file
        json.dump(final_data, final_result_file, indent=4, ensure_ascii=False)
        
        # Truncate the file to the current position (in case the new data is shorter than the old data)
        final_result_file.truncate()

def key_info_extraction(json_file_path, threshold, final_json):
    process_json_and_run_inference(json_file_path, threshold, final_json)
    