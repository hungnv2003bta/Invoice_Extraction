import json

# Path to your filtered dataset in JSON format
dataset_path = "/Users/hungnguyen/Developer/repos/tensorflow/Final/MC_OCR/mc_ocr/data/filtered_funsd_format.json"

# Define the mapping of ner_tags to actual labels
ner_tag_mapping = {
    0: "SELLER",
    1: "ADDRESS",
    2: "TIMESTAMP",
    3: "TOTAL_COST"
}

# Function to convert ner_tags from numeric to string labels
def convert_ner_tags(data):
    # Convert ner_tags using the predefined mapping
    data["ner_tags"] = [ner_tag_mapping[tag] for tag in data["ner_tags"]]
    return data

# Load the dataset
with open(dataset_path, "r", encoding="utf-8") as file:
    dataset = json.load(file)

# Convert ner_tags for each entry in the dataset
converted_dataset = [convert_ner_tags(entry) for entry in dataset]

# Save the converted dataset back to a JSON file
output_path = "/Users/hungnguyen/Developer/repos/tensorflow/Final/MC_OCR/mc_ocr/data/final_funsd_format.json"
with open(output_path, "w", encoding="utf-8") as file:
    json.dump(converted_dataset, file, indent=4, ensure_ascii=False)
