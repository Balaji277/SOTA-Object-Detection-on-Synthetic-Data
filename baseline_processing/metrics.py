def iou(boxA, boxB):
  
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou

def yeild_paths(paths):
  for p in paths:

     yield p

def get_bool_list(flag,any_list):
    flag_list = []
    for m in yeild_paths(any_list):
        
        if flag in m:
           flag_list.append("True")
        else:
           flag_list.append("False")
    return flag_list

def get_true_indices(flag_list):
    return [i for i,f in enumerate(flag_list) if "True" in f ]

def filtered_list_indices(index_list):
    return [i for i,f in enumerate(index_list) if "True" in f ]

def create_list_from_indices(any_list,index_list):
    k=[]
    for i in index_list:
       k.append(any_list[i])

    return k


def metrics(image_paths_pred, pred_boxes, pred_labels, image_paths_gt, gt_boxes, gt_labels, threshold):
    paths = np.unique(image_paths_pred)
    Tp,Fp= 0,0
    unique_labels= np.unique(gt_labels)
    for path in yeild_paths(paths):
        flag = "/".join(path.split("/")[-2:])
        pred_flag_list= get_bool_list(flag,image_paths_pred)
        gt_flag_list = get_bool_list(flag,image_paths_gt)
        
        pred_indices = get_true_indices(pred_flag_list)
        gt_indices = get_true_indices(gt_flag_list)

        pred_box_img = create_list_from_indices(pred_boxes,pred_indices)
        pred_labels_img = create_list_from_indices(pred_labels,pred_indices)
        gt_boxes_img = create_list_from_indices(gt_boxes,gt_indices)
        gt_labels_img = create_list_from_indices(gt_labels,gt_indices)

        predictions = zip(pred_box_img, pred_labels_img)
        ground_truth = zip(gt_boxes_img, gt_labels_img)
        
        for pred_b,pred_l in predictions:
            for gt_b,gt_l in ground_truth:
                 if isinstance(gt_l,float):
                    continue
                
                 if pred_l in gt_l:
                   if iou(pred_b,gt_b)>threshold:
                     Tp+=1
                   else:
                     Fp +=1

    precision = Tp/len(pred_labels)
    recall = Tp/ len(gt_labels)

    return precision, recall

def tpfp_metrics(image_paths_pred, pred_boxes, pred_labels, image_paths_gt, gt_boxes, gt_labels, threshold):
    paths = np.unique(image_paths_pred)
    Tp,Fp= 0,0
    unique_labels= np.unique(gt_labels)
    for path in yeild_paths(paths):
        flag = "/".join(path.split("/")[-2:])
        
        pred_flag_list= get_bool_list(flag,image_paths_pred)
        gt_flag_list = get_bool_list(flag,image_paths_gt)
        
        pred_indices = get_true_indices(pred_flag_list)
        gt_indices = get_true_indices(gt_flag_list)

        pred_box_img = create_list_from_indices(pred_boxes,pred_indices)
        pred_labels_img = create_list_from_indices(pred_labels,pred_indices)
        gt_boxes_img = create_list_from_indices(gt_boxes,gt_indices)
        gt_labels_img = create_list_from_indices(gt_labels,gt_indices)

        predictions = zip(pred_box_img, pred_labels_img)
        ground_truth = zip(gt_boxes_img, gt_labels_img)
        
        for pred_b,pred_l in predictions:
            for gt_b,gt_l in ground_truth:
                 if isinstance(gt_l,float):
                    break
                

                 if pred_l in gt_l:
                   if iou(pred_b,gt_b)>threshold:
                      Tp+=1
                   else:
                      Fp +=1

    return Tp, Fp

### Calculate Precision & Recall for All Models ###
import os
import numpy as np
import pandas as pd

if __name__=="__main__":
	root_dir = "/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists"
	metric_list = []
	model_name = "FasterRCNN"
	# Mount drive
	from google.colab import drive
	drive.mount('/content/drive')
	for f_csv in os.listdir(os.path.join(root_dir,"FinalExecution")):
    	    if model_name in f_csv:
       	       df_predictions = pd.read_csv(os.path.join(root_dir,"FinalExecution",f_csv))
       	       df_groundtruth = pd.read_csv(os.path.join(root_dir,"mAP_Prediction","Ground_Truth_Annotations_withoutindex.csv"))
       
       	       predboxes = df_predictions.iloc[:, 2:6].values.tolist() ## If index there then 2:6 , index not there 1:5
               predlabels = df_predictions["label"].values.tolist()
               image_path_pred = df_predictions["Path"].values.tolist()
               gt_boxes = df_groundtruth.iloc[:, 1:5].values.tolist()
               gt_labels = df_groundtruth["label"].values.tolist()
               image_path_gt = df_groundtruth["Path"].values.tolist()
       
               nmsthreshold = float(f_csv.split("_")[-1].split(".csv")[0][9:])
               scorethreshold = float(f_csv.split("_")[1][11:])
      
               precision, recall = metrics(image_path_pred , predboxes, predlabels, image_path_gt, gt_boxes, gt_labels,nmsthreshold)

               metric_list.append([model_name,nmsthreshold,scorethreshold,precision,recall])

               df = pd.DataFrame(metric_list,columns=["Model_Name","NMS Thresh","Score Thresh","Precision", "Recall"])
               df.to_csv(f"/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/mAP_Prediction/{model_name}.csv", index=None)