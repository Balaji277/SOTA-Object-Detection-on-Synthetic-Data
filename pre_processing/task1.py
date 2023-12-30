from utils import *
from tqdm import tqdm
import cv2

ROOT_DIR = '/content/drive/MyDrive/VLR_PROJECT'
input_annotations = f'{ROOT_DIR}/indoorCVPR_09_intersection/Annotations'
poster_path = f'{ROOT_DIR}/posters/augmented-samples-reduced'

# Collection of image paths
images_paths = get_all_image_paths_with_xml(input_annotations)
# Collection of annotation paths
annotation_paths = [image_path.replace('Images', 'Annotations').replace('.jpg','.xml') for image_path in images_paths]
# Collection of poster paths to be superimposed
posters = os.listdir(poster_path)

# Poster image object to corresponding categories in indoor CVPR dataset
object_to_category = {
    "poster-apple": ["apples", "apple"],
    "poster-camera": ["tv camera", "camera"],
    "poster-bowl": ["decorative bowl", "bowl", "large bowl", "fruit bowl", "bowl of fruit", "bowls", "Bowl"],
    "poster-cellphone": ["cell phone", "cell", "mobile", "mobile phone"],
    "poster-cup": ["paper cup", "cup", "coffee cup", "cup glass", "tea cup"],
    "poster-orange": ["orange", "Oranges"],
    "poster-water-bottle": ["water bottles", "seltzer bottle", "bottles", "bottle", "water bottle"]
}

def generate_images_with_poster(image_path, annotation_path, poster, poster_key):
  image = cv2.imread(image_path)
  boxes = get_boxes_from_xml(annotation_path)
  presence = False

  objects_present = set(boxes.keys())
  objects_to_check = set(object_to_category[poster_key])
  intersection = objects_present.intersection(objects_to_check)
  if len(intersection) != 0:
    presence = True

  non_overlapping_mask = find_non_overlapping_area(image, boxes)
  image_with_poster, poster_cs, status = place_poster(image, poster, non_overlapping_mask, boxes)
  image_name = image_path.split('/')[-1]
  sub_folder = image_path.split('/')[-2]
  parent_dir = f'{ROOT_DIR}/Images_with_posters/{poster_key}/{sub_folder}'
  if not os.path.exists(parent_dir):
    os.makedirs(parent_dir)
  cv2.imwrite(f'{ROOT_DIR}/Images_with_posters/{poster_key}/{sub_folder}/{image_name}', image_with_poster)
  return image_path, image_with_poster, poster_cs, presence

master_dict = {}

if __name__ == '__main__':
    for file_name in posters:
        object_name = file_name.split('.')[0]
        print(file_name)
        parent_dir = f'{ROOT_DIR}/Images_with_posters/{object_name}'
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        poster = cv2.imread(os.path.join(poster_path, file_name))
        master_dict[object_name] = []

        for image,annotation in tqdm(zip(images_paths, annotation_paths)):
            # print(image)
            try:
                image_path, image_with_poster, poster_cs, presence = generate_images_with_poster(image, annotation, poster, object_name)
                master_dict[object_name].append((image_path, poster_cs, presence))
            except:
                print(image)
                continue
    
    with open(f'{ROOT_DIR}/Dictionaries_and_lists/master_dict.json', 'w') as fp:
        json.dump(master_dict, fp)




    ### Correct master dictionary
    with open(f'{ROOT_DIR}/Dictionaries_and_lists/corrected_low_res.json', 'r') as fp:
        corrected_low_res = json.load(fp)

    with open(f'{ROOT_DIR}/Dictionaries_and_lists/corrected_missing_posters.json', 'r') as fp:
        corrected_missing_posters = json.load(fp)

    with open(f'{ROOT_DIR}/Dictionaries_and_lists/images_without_alterations.json', 'r') as fp:
        images_without_alterations = json.load(fp)

    ## combine all the dictionaries
    corrected_master_dict = {}
    for key in corrected_low_res.keys():
        corrected_master_dict[key] = corrected_low_res[key] + corrected_missing_posters[key] + images_without_alterations[key]

    with open(f'{ROOT_DIR}/Dictionaries_and_lists/corrected_master_dict.json', 'w') as fp:
        json.dump(corrected_master_dict, fp)