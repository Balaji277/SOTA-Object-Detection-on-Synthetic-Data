## Yolo ##

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


def Model(model, model_name, data_paths, labels, device, scores_thresh, nms_thresh):
    rows_list=[]

    for aug_img_path,label in zip(data_paths,labels):
        if not os.path.exists(f"/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/Task1b/{model_name}_scorethresh{scores_thresh}_nmsthresh{nms_thresh}"):

           os.mkdir(f"/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/Task1b/{model_name}_scorethresh{scores_thresh}_nmsthresh{nms_thresh}")

        if not os.path.exists(f"/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/Task1b/{model_name}_scorethresh{scores_thresh}_nmsthresh{nms_thresh}/"+label):
          #  import pdb; pdb.set_trace()
           os.mkdir(f"/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/Task1b/{model_name}_scorethresh{scores_thresh}_nmsthresh{nms_thresh}/"+label)
        save_dir = f"/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/Task1b/{model_name}_scorethresh{scores_thresh}_nmsthresh{nms_thresh}/"+label

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

            df1.to_csv(f"/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/Task1b/{model_name}_scorethresh{scores_thresh}_nmsthresh{nms_thresh}.csv", index=False)

            results.save(save_dir=os.path.join(save_dir,img_name))
          # print("Img without iou check saved:{}".format(file_name))
        except Exception:
            print(f"Following file had error:{img_name} with model-{model_name}, score_threshold-{scores_thresh}, nms_threshold-{nms_thresh}")


if __name__=="__main__":
	# Mount drive
	from google.colab import drive
	drive.mount('/content/drive')
	
	if not os.path.exists("/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/Task1b"):
   	   os.mkdir("/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/Task1b")
	
	Model_Name="YOLO" # FCOS, RetinaNet, SSD300, # FCOS, RetinaNet, SSD300,


	device="cuda" if torch.cuda.is_available() else "cpu"

	df = pd.read_csv("/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/original_to_aug_500.csv")
	data_paths = df["aImagePath"].values.tolist()
	labels = df["posterCategory"].values.tolist()
	df_prec = pd.read_csv("/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/mAP_Prediction/YOLO.csv")
	df_prec_new = df_prec.sort_values(by="Precision", ascending=False)
	nms_thresh = df_prec_new["NMS Thresh"].values.tolist()[0:3]
	score_thresh = df_prec_new["Score Thresh"].values.tolist()[0:3]
	model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

	for nms_thres, scr_thres in zip(nms_thresh,score_thresh):
    	    model.conf = scr_thres #Score thresh
            model.iou = nms_thres # nmsthresh
            Model(model, Model_Name, data_paths, labels, device, scr_thres, nms_thres)