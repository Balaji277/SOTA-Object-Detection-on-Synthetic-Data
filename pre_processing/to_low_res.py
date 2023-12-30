from utils import *
from tqdm import tqdm
import os
import json

if __name__ == '__main__':

    low_ratio_images = {}
    ROOT_DIR = '/content/drive/MyDrive/VLR_PROJECT'
    input_annotations = f'{ROOT_DIR}/indoorCVPR_09_intersection/Annotations'
    poster_path = f'{ROOT_DIR}/posters/augmented-samples-reduced'

    with open(f'{ROOT_DIR}/Dictionaries_and_lists/master_dict.json', 'r') as fp:
        master_dict = json.load(fp)

    object_to_category = {
        "poster-apple": ["apples", "apple"],
        "poster-camera": ["tv camera", "camera"],
        "poster-bowl": ["decorative bowl", "bowl", "large bowl", "fruit bowl", "bowl of fruit", "bowls", "Bowl"],
        "poster-cellphone": ["cell phone", "cell", "mobile", "mobile phone"],
        "poster-cup": ["paper cup", "cup", "coffee cup", "cup glass", "tea cup"],
        "poster-orange": ["orange", "Oranges"],
        "poster-water-bottle": ["water bottles", "seltzer bottle", "bottles", "bottle", "water bottle"]
    }

    corrected_low_res = {}
    images_without_alterations = {}
    err_cnt = []
    for key in master_dict.keys():
        corrected_low_res[key] = []
        for val in master_dict[key]:
            try:
                image_path = val[0]
                annotation_path = image_path.replace('.jpg','.xml').replace('Images', 'Annotations')
                orig_image = cv2.imread(image_path)
                poster_path = os.path.join(poster_path, key+'.jpg')
                poster = cv2.imread(poster_path)
                poster_cs = val[1]

                if poster_cs!=[0,0,0,0]:

                    boxes = get_boxes_from_xml(annotation_path)
                    presence = False

                    objects_present = set(boxes.keys())
                    objects_to_check = set(object_to_category[key])
                    intersection = objects_present.intersection(objects_to_check)
                    if len(intersection) != 0:
                        presence = True

                    xmin,ymin,xmax,ymax = poster_cs
                    poster_area = poster.shape[0]*poster.shape[1]
                    image_ref = orig_image
                    image_area = image_ref.shape[0]*image_ref.shape[1]
                    ratio = poster_area/image_area
                    if ratio < 0.01:
                        scale = 0.01/ratio
                        # if file_name == 'dining9.jpg':
                        #   import pdb;pdb.set_trace()
                        resized_poster = cv2.resize(poster,(0,0),fx=scale**0.5,fy=scale**0.5)
                        resized_shape = resized_poster.shape
                        corrected_low_res[key].append([image_path,[xmin,xmin+resized_shape[1],ymin,ymin+resized_shape[0]],presence])
                    
                    else:
                        images_without_alterations[key].append([image_path,poster_cs,presence])

            except:
                print(Exception)
                err_cnt.append(image_path)

    with open(f'{ROOT_DIR}/Dictionaries_and_lists/corrected_low_res.json', 'w') as fp:
        json.dump(corrected_low_res, fp)

    with open(f'{ROOT_DIR}/Dictionaries_and_lists/images_without_alterations.json', 'w') as fp:
        json.dump(images_without_alterations, fp)
