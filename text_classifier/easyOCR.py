# pip install vietocr

import matplotlib.pyplot as plt
from PIL import Image

from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

config = Cfg.load_config_from_name('vgg_seq2seq')

config['weights'] = 'weights/transformerocr.pth'
config['cnn']['pretrained']=False
config['device'] = 'cuda:0'

img = '/Users/hungnguyen/Developer/repos/tensorflow/Final/MC_OCR/mc_ocr/text_detector/crop_image_text/rotated_180.jpg_cropped_1.jpg'

result = Predictor(config)
img = Image.open(img)
plt.imshow(img)
s = result.predict(img)
print(s)
plt.show()
