#### All Models ######

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


def Model(model, model_name, weights, root, data_path, device, scores_thresh, nms_thresh):
  rows_list=[]
  for org_root,dirs,files in tqdm(os.walk(data_path)):
      for file in tqdm(files):
          try:
            dir_name = org_root.split("/")[-1]


            if not os.path.exists(os.path.join(root, dir_name)):
              os.mkdir(os.path.join(root, dir_name))


            img_path = os.path.join(org_root,file)
            img = read_image(img_path)
            img = img.to(device)
            file_name = file.split(".")[0]

            # Step 1: Initialize model with the best available weights
            model.eval()

            # Step 2: Initialize the inference transforms
            preprocess = weights.transforms()

            # Step 3: Apply inference preprocessing transforms
            batch = [preprocess(img)]

            # Step 4: Use the model and visualize the prediction
            prediction = model(batch)[0]
            if prediction["boxes"]==None:
               im.save(os.path.join(root,dir_name,file_name+".png"))

            # prediction["boxes"] = prediction["boxes"].to("cpu")
            labels = [weights.meta["categories"][i] for i in prediction["labels"]]
            box = draw_bounding_boxes(img, boxes=prediction["boxes"],labels=labels, colors="red", width=4, font_size=30)


            for i,label in enumerate(labels):
                dict1={"Path":img_path,"x_min":prediction["boxes"][i,0].detach().cpu().numpy(),"y_min":prediction["boxes"][i,1].detach().cpu().numpy(),"x_max":prediction["boxes"][i,2].detach().cpu().numpy(),"y_max":prediction["boxes"][i,3].detach().cpu().numpy(),"label":labels[i],"scores":prediction["scores"][i].detach().cpu().numpy()}
                rows_list.append(dict1)

            im = to_pil_image(box.detach())

            im.save(os.path.join(root,dir_name,file_name+".png"))
            # print("Img without iou check saved:{}".format(file_name))
          except Exception:
            print(f"Following file had error:{file_name} with model-{model_name}, score_threshold-{scores_thresh}, nms_threshold-{nms_thresh}")

          # Step5: Calculate Metrics
          df = pd.DataFrame(rows_list)
          df.to_csv(f"/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/FinalExecution/{model_name}_scorethresh{scores_thresh}_nmsthresh{nms_thresh}.csv")

if __name__=="__main__":
	if not os.path.exists("/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/FinalExecution"):
   	   os.mkdir("/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/FinalExecution")
	
	Model_Name="SSD300" # FCOS, RetinaNet, SSD300,
	data_path = "/content/drive/MyDrive/VLR_PROJECT/indoorCVPR_09_intersection/Images"


	root = "/content/drive/MyDrive/VLR_PROJECT/object_detection/" + Model_Name
	device="cuda" if torch.cuda.is_available() else "cpu"

	print(device)

	for hyperparams in ["nms_thresh","score_thresh"]: 
    	    for thresh in [0.01,0.05,0.1,0.2,0.3,0.5,0.7,0.9]: 
        	if hyperparams=="nms_thresh":
           	   thresh_str = str(thresh)
           	   if not os.path.exists(os.path.join(root,"score_thresh_0.5_"+"nms_thresh_"+thresh_str)):
              	      os.mkdir(os.path.join(root,"score_thresh_0.5_"+"nms_thresh_"+thresh_str))

           	   weights = SSD300_VGG16_Weights.DEFAULT # FCOS_ResNet50_FPN_Weights, RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT, SSD300_VGG16_Weights.DEFAULT, 
		   #FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, #FCOS_ResNet50_FPN_Weights.DEFAULT
           	   model = ssd300_vgg16(weights=weights, score_thresh=0.5, nms_thresh=thresh).to(device) # fcos_resnet50_fpn,retinanet_resnet50_fpn_v2, score_thresh=0.3, nms_thresh=0.1

                   Model(model, Model_Name, weights, os.path.join(root,"score_thresh_0.5_"+"nms_thresh_"+thresh_str), data_path, device, 0.5, thresh)

               if hyperparams=="score_thresh":
                  thresh_str = str(thresh)
           	  if not os.path.exists(os.path.join(root,"score_thresh_"+thresh_str+"_nms_thresh_0.5")):
              	     os.mkdir(os.path.join(root,"score_thresh_"+thresh_str+"_nms_thresh_0.5"))

           	  weights = SSD300_VGG16_Weights.DEFAULT # RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT, SSD300_VGG16_Weights.DEFAULT, #FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, 		  #FCOS_ResNet50_FPN_Weights.DEFAULT
           	  model = ssd300_vgg16(weights=weights, score_thresh=thresh, nms_thresh=0.5).to(device) # retinanet_resnet50_fpn_v2, score_thresh=0.3, nms_thresh=0.1

                  Model(model, Model_Name, weights, os.path.join(root,"score_thresh_"+thresh_str+"_nms_thresh_0.5"), data_path, device, thresh,0.5)