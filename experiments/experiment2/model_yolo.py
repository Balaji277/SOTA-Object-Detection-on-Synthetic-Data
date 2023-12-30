## YOLO Task2 ###


# Mount drive
from google.colab import drive
drive.mount('/content/drive')

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



def Model(model, model_name, path_generator, device, scores_thresh, nms_thresh):
    rows_list=[]
    for aug_img_path,label,scene in path_generator:
        if not os.path.exists(f"/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/Task2_FinalExecution" + f"/{model_name}_scoresthresh{scores_thresh}_nmsthresh{nms_thresh}"):
           os.mkdir(f"/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/Task2_FinalExecution" + f"/{model_name}_scoresthresh{scores_thresh}_nmsthresh{nms_thresh}")

        if not os.path.exists(f"/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/Task2_FinalExecution" + f"/{model_name}_scoresthresh{scores_thresh}_nmsthresh{nms_thresh}/"+label):
          #  import pdb; pdb.set_trace()
           os.mkdir(f"/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/Task2_FinalExecution" + f"/{model_name}_scoresthresh{scores_thresh}_nmsthresh{nms_thresh}/"+label)
        save_dir = f"/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/Task2_FinalExecution" + f"/{model_name}_scoresthresh{scores_thresh}_nmsthresh{nms_thresh}/"+label

        try:
            img = cv2.imread(aug_img_path)
            img_path = aug_img_path
            # img = img.to(device)

            img_name = img_path.split("/")[-1]
            # import pdb; pdb.set_trace()
            results = model(img)

            # prediction["boxes"] = prediction["boxes"].to("cpu")


            df = results.pandas().xyxy[0]
            image_path = [img_path] * len(df)

            df.insert(0,"Path",image_path, True)
            df.drop(columns=["class"])
            df.rename(columns={"xmin":"x_min","xmax":"x_max","ymin":"y_min","ymax":"y_max",'name': 'label', 'confidence': 'scores'}, inplace=True)
            df = df[["Path","x_min","y_min","x_max","y_max","label","scores"]]

            rows_list.append(df)

            df1 = pd.concat(rows_list)

            df1.to_csv(f"/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/Task2_FinalExecution" + f"/{model_name}_scorethresh{scores_thresh}_nmsthresh{nms_thresh}.csv", index=False)
            results.save(save_dir=os.path.join(save_dir,img_name))
        except Exception:
            print(f"Following file had error:{img_name} with model-{model_name}, score_threshold-{scores_thresh}, nms_threshold-{nms_thresh}")

def get_path(task_path):
    for root,dirs,files in os.walk(task_path):
        for f in files:
          if "poster-orange" not in root or "poster-cellphone" not in root:
              yield os.path.join(root,f),root.split("/")[-2], root.split("/")[-1]

if __name__=="__main__"

	Model_Name="YOLO"
	device="cuda" if torch.cuda.is_available() else "cpu"
	if not os.path.exists("/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/Task2_FinalExecution"):
   	   os.mkdir("/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/Task2_FinalExecution")
	


	root = "/content/drive/MyDrive/VLR_PROJECT/Task2"

	scr_thresh = 0.3
	nms_thresh= 0.5

	model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

	model.conf = scr_thresh #Score thresh
	model.iou = nms_thresh # nmsthresh
	Model(model, Model_Name, get_path(root), device, scr_thresh, nms_thresh)# score_thresh=0.05, nms_thresh=0.5,