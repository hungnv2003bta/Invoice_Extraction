# pip install vietocr
# RECOGNITION TEXT USING VIETOCR 
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import cv2
from PIL import Image
import os


config = Cfg.load_config_from_name('vgg_seq2seq')

# config['weights'] = 'weights/transformerocr.pth'
config['cnn']['pretrained'] = False
config['device'] = 'cpu'
detector = Predictor(config)


def crop_text_by_bboxes(image_name, rotated_image, bboxes, crop_image_folder):
  cropped_images = []
  recognized_texts = []
  
  for i, bbox in enumerate(bboxes):
    # Extract the coordinates from the bounding box
    x_min = min([point[0] for point in bbox])
    y_min = min([point[1] for point in bbox])
    x_max = max([point[0] for point in bbox])
    y_max = max([point[1] for point in bbox])
    
    # Crop the image using the bounding box coordinates
    cropped_image = rotated_image[y_min:y_max, x_min:x_max]
    cropped_images.append(cropped_image)
    
    # Save the cropped image
    cropped_image_pil = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

    text = detector.predict(cropped_image_pil)
    recognized_texts.append(text)
    cropped_image_pil.save(f"{crop_image_folder}/{image_name}_cropped_{i}.png")

  return recognized_texts



