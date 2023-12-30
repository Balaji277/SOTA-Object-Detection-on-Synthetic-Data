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

def Model(model, model_name, weights, data_path, labels, device, scores_thresh, nms_thresh):
  rows_list=[]
  for aug_img_path,label in zip(data_path,labels):
      if not os.path.exists(f"/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/Task1a/{model_name}_scoresthresh{scores_thresh}_nmsthresh{nms_thresh}"):
         os.mkdir(f"/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/Task1a/{model_name}_scoresthresh{scores_thresh}_nmsthresh{nms_thresh}")

      if not os.path.exists(f"/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/Task1a/{model_name}_scoresthresh{scores_thresh}_nmsthresh{nms_thresh}/"+label):
        #  import pdb; pdb.set_trace()
         os.mkdir(f"/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/Task1a/{model_name}_scoresthresh{scores_thresh}_nmsthresh{nms_thresh}/"+label)
      save_dir = f"/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/Task1a/{model_name}_scoresthresh{scores_thresh}_nmsthresh{nms_thresh}/"+label
      try:
          file_name = aug_img_path.split("/")[-1]

          img_path = aug_img_path
          img = read_image(img_path)
          img = img.to(device)

          # file_name = file_name.split(".")[0]

          # Step 1: Initialize model with the best available weights
          model.eval()

          # Step 2: Initialize the inference transforms
          preprocess = weights.transforms()

          # Step 3: Apply inference preprocessing transforms
          batch = [preprocess(img)]

          # Step 4: Use the model and visualize the prediction
          prediction = model(batch)[0]
          if prediction["boxes"]==None:
              im.save(os.path.join(save_dir,file_name))

          # prediction["boxes"] = prediction["boxes"].to("cpu")
          labels = [weights.meta["categories"][i] for i in prediction["labels"]]
          box = draw_bounding_boxes(img, boxes=prediction["boxes"],labels=labels, colors="red", width=4, font_size=30)


          for i,label in enumerate(labels):
              dict1={"Path":img_path,"x_min":prediction["boxes"][i,0].detach().cpu().numpy(),"y_min":prediction["boxes"][i,1].detach().cpu().numpy(),"x_max":prediction["boxes"][i,2].detach().cpu().numpy(),"y_max":prediction["boxes"][i,3].detach().cpu().numpy(),"label":labels[i],"scores":prediction["scores"][i].detach().cpu().numpy()}
              rows_list.append(dict1)

          im = to_pil_image(box.detach())

          im.save(os.path.join(save_dir,file_name))
          # print("Img without iou check saved:{}".format(file_name))
      except Exception:
          print(f"Following file had error:{file_name} with model-{model_name}, score_threshold-{scores_thresh}, nms_threshold-{nms_thresh}")

          # Step5: Calculate Metrics
      df = pd.DataFrame(rows_list)
      df.to_csv(f"/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/Task1a/{model_name}_scorethresh{scores_thresh}_nmsthresh{nms_thresh}.csv")


if __name__=="__main__":
	
	# Mount drive
	from google.colab import drive
	drive.mount('/content/drive')
	
	if not os.path.exists("/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/Task1a"):
   	   os.mkdir("/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/Task1a")
	
	Model_Name= "FasterRCNN" # FCOS, RetinaNet, SSD300,

	device="cuda" if torch.cuda.is_available() else "cpu"

	df = pd.read_csv("/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/original_to_aug_500.csv")
	data_paths = df["aImagePath"].values.tolist()
	labels = df["posterCategory"].values.tolist()
	df_prec = pd.read_csv("/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/mAP_Prediction/SSD300.csv")
	df_prec_new = df_prec.sort_values(by="Precision", ascending=False)
# import pdb; pdb.set_trace()
	weights = SSD300_VGG16_Weights.DEFAULT # FCOS_ResNet50_FPN_Weights, RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT, SSD300_VGG16_Weights.DEFAULT, #FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, #FCOS_ResNet50_FPN_Weights.DEFAULT
	model = ssd300_vgg16(weights=weights, score_thresh=0.5, nms_thresh=0.01).to(device) # fcos_resnet50_fpn,retinanet_resnet50_fpn_v2, score_thresh=0.3, nms_thresh=0.1
	nms_thresh = df_prec_new["NMS Thresh"].values.tolist()[2:]
	score_thresh = df_prec_new["Score Thresh"].values.tolist()[2:]
	for nms_thres, scr_thres in zip(nms_thresh,score_thresh):
    	    model = ssd300_vgg16(weights=weights, score_thresh= scr_thres, nms_thresh=nms_thres).to(device)
    	    Model(model, Model_Name, weights, data_paths, labels, device, scr_thres, nms_thres)