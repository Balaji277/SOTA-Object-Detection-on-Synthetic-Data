import cv2
import os
import shutil
import json

print("INSTRUCTIONS:\n\n")
print("Press k to keep image")
print("Press r to not keep image")
print("NOTE: THIS DOESN'T WORK IN COLAB")


def filter_folder(folder_dir, save_folder):
    saved = []

    if not os.path.isdir(folder_dir):
        print("ERROR: PHOTO FOLDER DOESN'T EXIST")
        exit(0)
    if not os.path.isdir(save_folder):
        print("ERROR: SAVE FOLDER DOESN'T EXIST")
        exit(0)


    #print(len(os.listdir(folder_dir)))
    #exit()

    '''
    with open(save_folder + "/filtered.json", "r") as jsonFile:
        data = json.load(jsonFile)
        print(len(data))
    exit()
    '''
    

    checkpoint = 0
    for photo_file in os.listdir(folder_dir):
        try:
            im = cv2.imread(folder_dir + "/" + photo_file)
            cv2.imshow("Image", im)
            while True:
                keep = cv2.waitKey(33)
                if keep == ord('k'):
                    saved.append(photo_file)
                    break
                elif keep == ord('r'):
                    break
            checkpoint +=1
            if checkpoint == 10:
                checkpoint = 0
                files = [folder_dir + "/" + s for s in saved]
                jsonFiles = json.dumps(files)
                with open(save_folder + "/filtered.json", "w") as outfile:
                    outfile.write(jsonFiles)
        except:
            print("FAILED FOR IMAGE: " + photo_file)

    files = [folder_dir + "/" + s for s in saved]
    jsonFiles = json.dumps(files)
    with open(save_folder + "/filtered.json", "w") as outfile:
        outfile.write(jsonFiles)
    
    #Use below if you want to save files to folder directly instead of list of files
    '''
    for s in saved:
        shutil.copy(folder_dir + "/" + s, save_folder + "/" + s)
    '''
    
    return saved


#Folder for photos to filter
photo_folder = "corridor"

#Folder for where to save the json of filtered files to
#If you uncomment section in code, also saves files to this folder
save_folder = "corridor_filtered"

saved = filter_folder(photo_folder, save_folder)

#print(saved)
    
    
