import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###############################################################################################
# Dataframes pre processing
###############################################################################################

# Pre processing
def trim_path(path):
  return "/".join(path.split('/')[-2:])

def pre_process(df):
  return_df= df
  # 2. Update path for indexing
  return_df['Path'] = return_df['Path'].apply(lambda path: "/".join(path.split('/')[-2:]))
  # 3. Combine in one column
  return_df['oBbox'] = return_df[["x_min", "y_min", "x_max", "y_max", "label"]].values.tolist()
  # 4. Drop unrequired columns
  return_df.drop(columns=["x_min", "y_min", "x_max", "y_max", "label"], axis=1, inplace=True)
  # 5. Rename path column to highlight original predictions
  return_df.rename(columns={"Path": "oImagePath"}, inplace=True)

  return return_df

def group_predictions(df):
  return_df = df.groupby(by=["oImagePath"])['oBbox'].agg(list).reset_index()
  # return_df['o_bbox'] = return_df['o_bbox'].apply(lambda x: sorted(x))
  return return_df

###############################################################################################
# Robustness score calculation utilities
###############################################################################################


def is_inside(boxA, boxB):
  within_x = (boxA[0]<=boxB[0] and boxA[0]<=boxB[2] and boxA[2]>=boxB[0] and boxA[2]>=boxB[2])
  within_y = (boxA[1]<=boxB[1] and boxA[1]<=boxB[3] and boxA[3]>=boxB[1] and boxA[3]>=boxB[3])
  return within_x and within_y

def iou(boxA, boxB):
  # Check if inside
  if (is_inside(boxA, boxB) or is_inside(boxB, boxA)):
    # either boxA is inside boxB or opposite
    return 1.0
	# determine the (x, y)-coordinates of the intersection rectangle
  # Reference: https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
  interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
  boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
  iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
  return iou


# Map
object_to_category = {
    "apple": ["apples", "apple"],
    "camera": ["tv camera", "camera"],
    "bowl": ["decorative bowl", "bowl", "large bowl", "fruit bowl", "bowl of fruit", "bowls", "Bowl"],
    "cellphone": ["cell phone", "cell", "mobile", "mobile phone"],
    "cup": ["paper cup", "cup", "coffee cup", "cup glass", "tea cup"],
    "orange": ["orange", "Oranges"],
    "water-bottle": ["water bottles", "seltzer bottle", "bottles", "bottle", "water bottle"]
}
# Get score using poster coordinates
def is_detected(poster_coordinates, poster_label, aug_coordinates, aug_label, iou_threshold):
  # print(iou(aug_coordinates, poster_coordinates))
  if iou(aug_coordinates, poster_coordinates)>iou_threshold:
    return aug_label in object_to_category[poster_label]
  return False


def calculate_robustness_score(poster_category, poster_coordinates, augmented_boxes, iou_threshold):
  for aug_box in augmented_boxes:
    if is_detected(poster_coordinates, poster_category, aug_box[:-1], aug_box[-1], iou_threshold):
      return 1
  return 0

def get_robustness_score(row, iou_threshold):
  # cast poster coordinates to float
  poster_coord = [float(x) for x in row['posterCoordinates'].strip('[]').split(',')]
  return calculate_robustness_score(row['posterCategory'], poster_coord, row['aBbox'], iou_threshold)
  
  
###############################################################################################
# Utilities for plots
###############################################################################################

def plot_and_save_distributions(distribution, figure_location):
    """
        E.g.: {"FasterRCNN": {"apple": 6, "bowl": 4, "cup": 0, "water-bottle": 0}, "FCOS": {"apple": 8, "bowl": 2, "cup": 2, "water-bottle": 0}, "RetinaNet": {"apple": 0, "bowl": 0, "cup": 0, "water-bottle": 0}, "SSD300": {"apple": 0, "bowl": 0, "cup": 0, "water-bottle": 0}, "YOLO": {"apple": 0, "bowl": 0, "cup": 0, "water-bottle": 0}}
    """
    df = pd.DataFrame.from_dict(distribution).transpose()
    colors = ['Red', 'Blue', 'Green', 'Orange','Purple']
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_width = 0.2  # Adjust the width of the bars based on preference
    bar_positions = np.arange(len(df.columns))

    for i, region in enumerate(df.index):
        values = df.loc[region]
        ax.bar(bar_positions + i * bar_width, values, bar_width, color=colors[i], label=region)

    ax.set_xticks(bar_positions + (len(df.index) - 1) * bar_width / 2)
    ax.set_xticklabels(df.columns)
    ax.set_xlabel('Poster category')
    ax.set_ylabel('Number of positive poster object detection')
    ax.legend(title='Object Detection Models')
    ax.set_title("Poster Object Detection Distribution Across Models")

    plt.savefig(figure_location)

    plt.show()
    
    
# Params
model_name = "YOLO" #SSD300, RetinaNet, FasterRCNN, FCOS, YOLO
scores_thresh = 0.2
nms_thresh = 0.5

file_path = f"./VLR_PROJECT/Dictionaries_and_lists/FinalExecution/{model_name}_scorethresh{scores_thresh}_nmsthresh{nms_thresh}.csv"

# Inputs
random_100_picks = "./VLR_PROJECT/Dictionaries_and_lists/random_100picks.csv"
original_to_aug_500 = './VLR_PROJECT/Task2_images_with_augmented_image_paths_and_poster_coordinates.csv'

# Output/Results
orig_model_results = f"./VLR_PROJECT/Dictionaries_and_lists/FinalExecution/{model_name}_scorethresh{scores_thresh}_nmsthresh{nms_thresh}.csv"
aug_model_results = f"./VLR_PROJECT/Dictionaries_and_lists/Task2_FinalExecution/{model_name}_scorethresh{scores_thresh}_nmsthresh{nms_thresh}.csv"

# Plot figures
figure_location = f"/content/drive/MyDrive/VLR_PROJECT/figures/{model_name}_score{scores_thresh}_nms{nms_thresh}.png"

