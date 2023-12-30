## YOLO ##

import json
import os
import torch
from torchvision.io.image import read_image
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights, ssd300_vgg16, SSD300_VGG16_Weights, fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights, FCOS_ResNet50_FPN_Weights, fcos_resnet50_fpn
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
import cv2
import pandas as pd
from tqdm import tqdm

if not os.path.exists("/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/FinalExecution"):
   os.mkdir("/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/FinalExecution")

def Model(model, model_name, root, data_path, device, scores_thresh, nms_thresh):
  rows_list=[]
  for org_root,dirs,files in tqdm(os.walk(data_path)):
      for file in tqdm(files):
          try:
            dir_name = org_root.split("/")[-1]


            if not os.path.exists(os.path.join(root, dir_name)):
              os.mkdir(os.path.join(root, dir_name))


            img_path = os.path.join(org_root,file)
            img = cv2.imread(img_path)
            # img = img.to(device)
            file_name = file.split(".")[0]

            results = model(img)

            df = results.pandas().xyxy[0]
            image_path = [img_path] * len(df)

            df.insert(0,"Path",image_path, True)
            df.drop(columns=["class"])
            df.rename(columns={"xmin":"x_min","xmax":"x_max","ymin":"y_min","ymax":"y_max",'name': 'label', 'confidence': 'scores'}, inplace=True)
            df = df[["Path","x_min","y_min","x_max","y_max","label","scores"]]

            rows_list.append(df)

            df1 = pd.concat(rows_list)

            df1.to_csv(f"/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/FinalExecution/{model_name}_scorethresh{scores_thresh}_nmsthresh{nms_thresh}.csv", index=False)

            results.save(save_dir=os.path.join(root,dir_name,file_name))

          except Exception:
            print(f"Following file had error:{file_name} on path {img_path} with model-{model_name}, score_threshold-{scores_thresh}, nms_threshold-{nms_thresh}")

if __name__=="__main__":

	Model_Name="YOLO" 
	data_path = "/content/drive/MyDrive/VLR_PROJECT/indoorCVPR_09_intersection/Images"


	root = "/content/drive/MyDrive/VLR_PROJECT/object_detection/" + Model_Name
	device="cuda" if torch.cuda.is_available() else "cpu"

	
	# Mount drive
	from google.colab import drive
	drive.mount('/content/drive')

	for hyperparams in ["nms_thresh","score_thresh"]:
    	    for thresh in [0.01,0.05,0.1,0.2,0.3,0.5,0.7,0.9]: 
        	if hyperparams=="nms_thresh":
           	   thresh_str = str(thresh)
           	   if not os.path.exists(os.path.join(root,"score_thresh_0.5_"+"nms_thresh_"+thresh_str)):
              	      os.mkdir(os.path.join(root,"score_thresh_0.5_"+"nms_thresh_"+thresh_str))

           	   model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
                   model.conf = 0.5
                   model.iou = thresh
                   Model(model, Model_Name, os.path.join(root,"score_thresh_0.5_"+"nms_thresh_"+thresh_str), data_path, device, 0.5, thresh)

        	if hyperparams=="score_thresh":
                   thresh_str = str(thresh)
                   if not os.path.exists(os.path.join(root,"score_thresh_"+thresh_str+"_nms_thresh_0.5")):
                      os.mkdir(os.path.join(root,"score_thresh_"+thresh_str+"_nms_thresh_0.5"))

           	   model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
                   model.conf = thresh
                   model.iou = 0.5
                   Model(model, Model_Name, os.path.join(root,"score_thresh_"+thresh_str+"_nms_thresh_0.5"), data_path, device, thresh,0.5)