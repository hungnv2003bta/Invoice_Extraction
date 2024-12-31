from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches

data = {
    "id": "mcocr_public_145014mjzwg",
        "words": [
            "VinCommerce",
            "VM+ QNH Tổ 7, khu Minh Tiến A",
            "Tổ 7, Khu Minh Tiến A",
            "P. Cẩm Bình, TP. Cẩm Phả, QNH",
            "Ngày bán: 13/08/2020 19:12",
            "TỔNG TIỀN PHẢI T.TOÁN",
            "62.600"
        ],
        "bboxes": [
            [
                108,
                116,
                226,
                151
            ],
            [
                261,
                76,
                438,
                101
            ],
            [
                286,
                104,
                413,
                122
            ],
            [
                259,
                124,
                437,
                148
            ],
            [
                103,
                211,
                332,
                235
            ],
            [
                95,
                579,
                292,
                606
            ],
            [
                410,
                587,
                480,
                610
            ]
        ],
        "ner_tags": [
            0,
            1,
            1,
            1,
            2,
            3,
            3
        ],
        "image_path": "/Users/hungnguyen/Developer/repos/tensorflow/Final/MC_OCR/mc_ocr/data/mc_ocr_train_filtered/mcocr_public_145014mjzwg.jpg"
}

def visualize_data(data):
    image_path = data["image_path"]
    words = data["words"]
    bboxes = data["bboxes"]

    # Load image
    image = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Plot bounding boxes and words
    for bbox, word in zip(bboxes, words):
        x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.text(x, y, word, fontsize=12, color='blue')

    plt.show()

visualize_data(data)