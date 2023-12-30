from utils import *
from tqdm import tqdm
import os,random,json,cv2


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

    corrected_missing_posters = {}
    err_cnt = []
    skip = []
    for key in master_dict.keys():
        corrected_missing_posters[key] = []
        for val in master_dict[key]:
            try:
                image_path = val[0]
                annotation_path = image_path.replace('.jpg','.xml').replace('Images', 'Annotations')
                orig_image = cv2.imread(image_path)
                poster_path = os.path.join(poster_path, key+'.jpg')
                poster = cv2.imread(poster_path)
                poster_cs = val[1]

                if poster_cs==[0,0,0,0]:

                    boxes = get_boxes_from_xml(annotation_path)
                    presence = False

                    objects_present = set(boxes.keys())
                    objects_to_check = set(object_to_category[key])
                    intersection = objects_present.intersection(objects_to_check)
                    if len(intersection) != 0:
                        boxes_n = {inter:boxes[inter] for inter in intersection}
                        # import pdb;pdb.set_trace()
                        non_overlapping_mask = find_non_overlapping_area(orig_image, boxes_n)
                        image_with_poster, poster_cs, status = place_poster(orig_image, poster, non_overlapping_mask, boxes_n)
                        presence = True
                    
                    else:                        
                        # import pdb;pdb.set_trace()
                        poster_h, poster_w,_ = poster.shape
                        h, w, _ = orig_image.shape
                        x_start, y_start = random.randint(0,poster_w), random.randint(0,poster_h)
                        ctr = 0
                        while not is_valid(x_start, y_start, poster_w, poster_h, w, h):
                            x_start, y_start = random.randint(0,poster_w), random.randint(0,poster_h)
                            ctr+=1
                            if ctr>20:
                                skip.append(image_path)
                                break

                        image_with_poster = orig_image.copy()
                        if ctr>20:
                            poster_cs = [0,0,0,0]
                            presence = 'poster larger than image'

                        else:
                            image_with_poster[y_start:y_start + poster_h, x_start:x_start + poster_w] = poster
                            poster_cs = [x_start, y_start, x_start + poster_w, y_start + poster_h]
                            presence = False

                        image_name = image_path.split('/')[-1]
                        path_to_save = f'{ROOT_DIR}/posters/poster_cs_zeros_solved/{key}/{image_name}'
                        cv2.imwrite(path_to_save,image_with_poster)
                        corrected_missing_posters[key].append([image_path,poster_cs,presence])

            except:
                print(Exception)
                err_cnt.append(image_path)

    with open(f'{ROOT_DIR}/Dictionaries_and_lists/corrected_missing_posters.json', 'w') as fp:
        json.dump(corrected_missing_posters, fp)

