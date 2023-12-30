import cv2
import os
import shutil
import json
import glob 

#loc_dict = {}
loc_dict = json.load(open("wall_loc_backup_dentaloffice_office3.json"))

def set_wall_loc(event, x, y, flags, params): 
    new_im_path = params[1]
    if event == cv2.EVENT_LBUTTONDOWN: 
        (params[0])[new_im_path] = (x, y)
        print(x,y)
        params[2] = 1

foundCheckpoint = False
for folder in sorted(glob.glob('*_filtered')): 
    foundCheckpoint = foundCheckpoint or folder == "dentaloffice_filtered"
    if foundCheckpoint:
        with open(folder + "/filtered.json", "r") as jsonFile:
            data = json.load(jsonFile)
            for im_path in data:
                print(folder, im_path)
                im = cv2.imread(im_path)
                cv2.imshow("Image", im)
                params = [loc_dict, im_path, 0]
                cv2.setMouseCallback('Image', set_wall_loc, params) 
                while params[2] < 1:    
                    cv2.waitKey(10)
                cv2.destroyAllWindows()
        with open("wall_loc.json", "w") as outfile: 
            json.dump(loc_dict, outfile)
                    
                 
    
