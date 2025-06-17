#!/bin/bash

# Step 1: Install base requirements from requirements.txt
pip install -r requirements.txt

# Step 2: Install additional dependencies
pip install vietocr --no-deps
pip install prefetch-generator==1.0.1
pip install albumentations==1.4.2
pip install einops==0.2.0
pip install gdown==4.4.0
pip install pillow==10.2.0
