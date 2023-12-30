# Variability in Performance of SOTA models in Object Detection towards Synthetic Dataset

## Setup
```bash
pip install -r requirements.txt
# or 
python3 -m pip install -r requirements.txt
```
```bash
# Each files are independent and can be executed using 
python3 <file_name>.py
```

## Dataset
Dataset used for this study is from [IndoorCVPR](https://web.mit.edu/torralba/www/indoor.html)

Out of the indoor scenes were picked:
1. bedrooms
2. dining rooms
3. living rooms
4. lobbies
5. meeting rooms
6. offices
7. restaurants

The images from these scenes were augmented with posters containing following 4 objects:
1. Apple
2. Cup
3. Waterbottle
4. Bowl

## Program
Initial code was written in Google Colab. The following code base describes the important code pieces used. On a high level, the code is structured in three parts.
1. pre_processing: This module contains code used to pre-process the images including but not limited to, picking above mentioned indoor scenes, removing images with errorneous or missing annotations, etc.
2. baseline_processing : This conatins code used for inital baseline study predictions without any augmentations on actual image. Models are divided into 2 separate python files: 'others' containing code for Faster-RCNN, FCOS, SSD300, and RetinaNet and 'yolo' as separate file because of difference in implementation.
3. experiments: This contains each of the three experiments as described in paper. Starting from expriment 1 containing code for programmatic placement of poster on original image, experiment 2 containing code when base image contained the poster object, and finally experiment 3 contains code around warping and warping combined with belnding poster images on base image.
4. metrics.py: This file contains code for calculating the robustness score given base bounding box predictions with labels, augmented poster image coordinates, and bounding boxes with corresponding labels on poster augmented image.

## Compute Resources 
Instance types from Google Colab: A100 and T4 GPU

# Authors
1. Alexander Lyons(alyons)
2. Mangalam Sahai (msahai)
3. Balaji Praneeth Boga (bboga)
4. Niranjan Kumawat (nkumawat)