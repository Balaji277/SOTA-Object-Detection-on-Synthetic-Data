# Retrieve images with object similar to one in poster image
from utils import *
from tqdm import tqdm
import json

ROOT_DIR = '/content/drive/MyDrive/VLR_PROJECT'
input_annotations = f'{ROOT_DIR}/indoorCVPR_09_intersection/Annotations'
poster_path = f'{ROOT_DIR}/posters/augmented-samples-reduced'

with open(f'{ROOT_DIR}/Dictionaries_and_lists/corrected_master_dict.json', 'r') as fp:
    corrected_master_dict = json.load(fp)

task2_dict = {}
for key in corrected_master_dict.keys():
    task2_dict[key] = []
    for val in corrected_master_dict[key]:
        if val[2] == True:
            task2_dict[key].append(val)

with open(f'{ROOT_DIR}/Dictionaries_and_lists/task2_dict.json', 'w') as fp:
    json.dump(task2_dict, fp)    