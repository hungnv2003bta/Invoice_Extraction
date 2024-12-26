import os

CONFIG_ROOT = os.path.dirname(__file__)
OUTPUT_ROOT = '/Users/hungnguyen/Developer/repos/tensorflow/Final/dataset/test_output2'


def full_path(sub_path, file=False):
    path = os.path.join(CONFIG_ROOT, sub_path)
    if not file and not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            print('full_path. Error makedirs',path)
    return path


def output_path(sub_path):
    path = os.path.join(OUTPUT_ROOT, sub_path)
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except:
            print('output_path. Error makedirs',path)
    return path

gpu = '0'  # None or 0,1,2...
dataset = 'mc_ocr_train_filtered'

# input data from organizer
raw_train_img_dir = full_path('data/mc_ocr_train')
raw_img_dir=full_path('data/{}'.format(dataset))
raw_csv = full_path('data/mcocr_train_df.csv', file=True)

# EDA
json_data_path = full_path('EDA/final_data.json', file=True)
filtered_train_img_dir=full_path('data/mc_ocr_train_filtered')
filtered_csv = full_path('data/mcocr_train_df_filtered.csv', file=True)

# # key information
# kie_visualize = True
# kie_model = full_path('key_info_extraction/PICK/saved/models/PICK_Default/test_0121_212713/model_best.pth', file=True)
# kie_boxes_transcripts = output_path('key_info_extraction/{}/boxes_and_transcripts'.format(dataset))
# kie_out_txt_dir = output_path('key_info_extraction/{}/txt'.format(dataset))
# kie_out_viz_dir = output_path('key_info_extraction/{}/viz_imgs'.format(dataset))
