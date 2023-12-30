import cv2, os, xmltodict, json, random
import xmltodict, json, random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Constants
ROOT_DIR = '/content/drive/MyDrive/VLR_PROJECT'

"""##Surface normals and depth map"""

### Obtain Depth Maps ###
import os
import timm
## MiDas Model
import cv2
import torch
import urllib.request
import json

import matplotlib.pyplot as plt
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

filtered_dict_folders = "/content/drive/MyDrive/VLR_PROJECT/indoorCVPR_09_filtered/"

filtered_images = []
for folder in sorted(os.listdir(filtered_dict_folders)):
  print(folder)
  filtered_room = json.load(open(os.path.join(filtered_dict_folders, folder, "filtered.json"), "r"))
  for im_path in filtered_room:
    filtered_images.append(os.path.join("/content/drive/MyDrive/VLR_PROJECT/indoorCVPR_09/Images", im_path))

print(len(filtered_images))

i = 0
for file in filtered_images:
    try:
      img = cv2.imread(file)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

      input_batch = transform(img).to(device)

      with torch.no_grad():
          prediction = midas(input_batch)

          prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
          ).squeeze()
      output = prediction.cpu().numpy()
      name = file.split("/")[-2] + "_" + file.split("/")[-1]
      plt.imsave(os.path.join("/content/drive/MyDrive/VLR_PROJECT/depth_maps_filtered/", "depth_map_" + name),output)
      print(i)
      i+=1
    except:
      print("FAILED on " + str(i))

#### Surface Normals Function ###
import cv2
import numpy as np
from PIL import Image
import numpy as np

def calculate_normals(depth_map, focal_length):
    height, width = depth_map.shape

    # Calculate gradients of the depth map
    dx, dy = np.gradient(depth_map)

    # Calculate the normal vectors
    normals = np.zeros((height, width, 3))
    normals[..., 0] = -dx / focal_length
    normals[..., 1] = -dy / focal_length
    normals[..., 2] = 1.0

    # Normalize the normals
    norm = np.sqrt(np.sum(normals ** 2, axis=2, keepdims=True))
    normals /= norm

    return normals

### Surface Normals Dictionary #####
import os
import matplotlib.pyplot as plt
surface_normals = {}
test_images = [((img.split("/")[-1]).split(".")[0])[10:] for img in os.listdir("/content/drive/MyDrive/VLR_PROJECT/depth_maps_filtered")]
# import pdb; pdb.set_trace()
for im in test_images:
  depth_image = np.array(Image.open('/content/drive/MyDrive/VLR_PROJECT/depth_maps_filtered/depth_map_' + im + ".jpg"))

  depth_image = cv2.cvtColor(depth_image,cv2.COLOR_BGR2GRAY)

  focal_length = 1
  principal_point = (depth_image.shape[1] / 2, depth_image.shape[0] / 2)

  normalized_depth = depth_image / focal_length

  surface_normals[im] = calculate_normals(normalized_depth, focal_length)

def visualize_normals(normals):
    # Map the normal vectors to RGB colors
    colors = (normals + 1) * 0.5 * 255
    colors = colors.astype(np.uint8)

    return colors

normal_images = {}
for k, v in surface_normals.items():
  normal_images[k] = visualize_normals(v)

"""Superimposing image (warping)"""

## Get Average Normals at a Point Loc ######

from numpy import linalg as LA

def average_normal(normal, poster_size, loc):
    loc_y, loc_x = loc
    check_range = 30

    sub_normals = normal[
        int(loc_y-check_range/2):
        int(loc_y+check_range/2),
        int(loc_x-check_range/2):
        int(loc_x+check_range/2)]

    sub_normals = sub_normals.reshape(-1,3)

    sub_normals = sub_normals[~np.all(sub_normals == np.array([0,0,1]),axis=-1)]
    sub_normals = sub_normals[~np.all(sub_normals == np.array([0,0,-1]),axis=-1)]

    avg = np.mean(sub_normals,axis=0)

    return avg

## Calculate the plane to put the poster on ##

def calculateNormalPlane(normal):

  normal/= np.linalg.norm(normal)

  #Calculate Angle of normal to vertical axis
  angle_to_vertical = np.pi/2-np.arcsin(abs(normal[1]))

  #Get the perpendicular vector on other side of vertical axis
  norm = np.sqrt(normal[0]**2 + normal[2]**2)/np.max([np.cos(angle_to_vertical), 0.000000001])
  perp_vec = np.array([-normal[0]/norm, np.sin(angle_to_vertical), -normal[2]/norm]).reshape(3)

  #Get third perpendicular vector
  perp_vec_2 = np.cross(normal, perp_vec)

  basis = np.array([perp_vec_2, perp_vec, normal]).T

  return basis

## Warping function ##

def poster_warp(poster_img, normal, loc, poster_size, room):

  height, width, _ = poster_img.shape

  #Get the average normal
  normal_vec = average_normal(normal, poster_size, loc)

  loc_y, loc_x = loc

  x, y = np.meshgrid(np.arange(width), np.arange(height))
  y = y.astype(np.float32) - height/2
  x = x.astype(np.float32) - width/2
  coords = np.dstack((x, y, np.zeros((height, width))))
  coords = coords.reshape(coords.shape[0]*coords.shape[1],coords.shape[2]).T

  extrinsics = calculateNormalPlane(normal_vec)

  #Assumed intrinsic parameters
  f = 1000
  cx = 0
  cy = 0
  intrinsics = np.array([[f,0,cx],[0,f,cy],[0,0,1]])

  m = intrinsics @ extrinsics

  #Rotate the poster
  warped_coords = (m @ coords).T


  #Arbitrarily change depth for camera
  warped_coords[:,2] += 10000

  #Project points onto plane
  warped_coords = warped_coords[:, 0:2]/warped_coords[:,2,None]


  #Align poster
  min_x = np.min(warped_coords[:, 0])
  min_y = np.min(warped_coords[:, 1])
  warped_coords[:, 0] -= min_x
  warped_coords[:, 1] -= min_y

  #warped_coords = np.round(warped_coords).astype(np.int32)
  max_x = np.max(warped_coords[:, 0])
  max_y = np.max(warped_coords[:, 1])
  max_all = max(max_x, max_y)

  #Extract the colors of the poster
  poster = np.zeros((poster_size, poster_size, 4)).astype(poster_img.dtype)
  colors = poster_img.reshape(poster_img.shape[0]*poster_img.shape[1], poster_img.shape[2])


  for i in range(warped_coords.shape[0]):
      y_sample = round((poster_size-1) *  warped_coords[i, 1]/max_all)
      x_sample = round((poster_size-1) * warped_coords[i, 0]/max_all)

      poster[y_sample, x_sample, :3] = colors[i]
      poster[y_sample, x_sample, 3] = 1

  new_size = int(np.floor((np.min(room.shape[0:2])/6)/20)*20)
  poster = cv2.resize(poster, (new_size, new_size))
  poster = np.dstack((poster, np.ones_like(poster[:,:,0])))
  poster_warped = np.copy(poster[:,:,:3])

  y_ind_min = max(int(loc_y - poster.shape[0]/2), 0)
  y_ind_max = min(int(loc_y + poster.shape[0]/2), room.shape[0])
  x_ind_min = max(int(loc_x - poster.shape[1]/2), 0)
  x_ind_max = min(int(loc_x + poster.shape[1]/2), room.shape[1])

  #Add poster to room
  room[y_ind_min:y_ind_max, x_ind_min:x_ind_max] -= room[y_ind_min:y_ind_max, x_ind_min:x_ind_max,:] * poster[:,:,3,None]
  room[y_ind_min:y_ind_max, x_ind_min:x_ind_max] += poster[:,:,0:3] * poster[:,:,3,None]

  poster_mask = np.zeros_like(room)
  poster_mask[y_ind_min:y_ind_max, x_ind_min:x_ind_max] += poster[:,:,3,None]

  poster_alone = np.zeros_like(room)
  poster_alone[y_ind_min:y_ind_max, x_ind_min:x_ind_max] += poster[:,:,0:3] * poster[:,:,3,None]

  return room, poster_warped, poster_mask, poster_alone

## Warping function for when don't actually want to warp ##

def no_poster_warp(poster_img, normal, loc, poster_size, room):

  height, width, _ = poster_img.shape

  #Get the average normal

  loc_y, loc_x = loc

  x, y = np.meshgrid(np.arange(width), np.arange(height))
  y = y.astype(np.float32) - height/2
  x = x.astype(np.float32) - width/2
  coords = np.dstack((x, y, np.zeros((height, width))))
  coords = coords.reshape(coords.shape[0]*coords.shape[1],coords.shape[2]).T

  extrinsics = np.eye(3)

  #Assumed intrinsic parameters
  f = 1000
  cx = 0
  cy = 0
  intrinsics = np.array([[f,0,cx],[0,f,cy],[0,0,1]])

  m = intrinsics @ extrinsics
  #print(m.shape)


  #Rotate the poster
  warped_coords = (m @ coords).T


  #Arbitrarily change depth for camera
  warped_coords[:,2] += 10000

  #Project points onto plane
  warped_coords = warped_coords[:, 0:2]/warped_coords[:,2,None]


  #Align poster
  min_x = np.min(warped_coords[:, 0])
  min_y = np.min(warped_coords[:, 1])
  warped_coords[:, 0] -= min_x
  warped_coords[:, 1] -= min_y

  #warped_coords = np.round(warped_coords).astype(np.int32)
  max_x = np.max(warped_coords[:, 0])
  max_y = np.max(warped_coords[:, 1])
  max_all = max(max_x, max_y)

  #Extract the colors of the poster
  poster = np.zeros((poster_size, poster_size, 4)).astype(poster_img.dtype)
  colors = poster_img.reshape(poster_img.shape[0]*poster_img.shape[1], poster_img.shape[2])


  for i in range(warped_coords.shape[0]):
      y_sample = round((poster_size-1) *  warped_coords[i, 1]/max_all)
      x_sample = round((poster_size-1) * warped_coords[i, 0]/max_all)

      poster[y_sample, x_sample, :3] = colors[i]
      poster[y_sample, x_sample, 3] = 1


  new_size = int(np.floor((np.min(room.shape[0:2])/6)/20)*20)
  poster = cv2.resize(poster, (new_size, new_size))
  poster = np.dstack((poster, np.ones_like(poster[:,:,0])))
  poster_warped = np.copy(poster[:,:,:3])

  y_ind_min = max(int(loc_y - poster.shape[0]/2), 0)
  y_ind_max = min(int(loc_y + poster.shape[0]/2), room.shape[0])
  x_ind_min = max(int(loc_x - poster.shape[1]/2), 0)
  x_ind_max = min(int(loc_x + poster.shape[1]/2), room.shape[1])

  #Add poster to room
  room[y_ind_min:y_ind_max, x_ind_min:x_ind_max] -= room[y_ind_min:y_ind_max, x_ind_min:x_ind_max,:] * poster[:,:,3,None]
  room[y_ind_min:y_ind_max, x_ind_min:x_ind_max] += poster[:,:,0:3] * poster[:,:,3,None]

  poster_mask = np.zeros_like(room)
  poster_mask[y_ind_min:y_ind_max, x_ind_min:x_ind_max] += poster[:,:,3,None]

  poster_alone = np.zeros_like(room)
  poster_alone[y_ind_min:y_ind_max, x_ind_min:x_ind_max] += poster[:,:,0:3] * poster[:,:,3,None]

  return room, poster_warped, poster_mask, poster_alone


def load_image(path):
    img = cv2.imread(path)
    return img

# Apply augmentation to poster object
def augmentPoster(angle, flip):
  poster_obj = np.copy(poster[25:75, 15:65])

  center = (poster_obj.shape[0]/2,poster_obj.shape[1]/2)
  R = cv2.getRotationMatrix2D(center, angle, 1.0)
  augmented = cv2.warpAffine(poster_obj, R, (poster_obj.shape[0],poster_obj.shape[1]), borderValue=(205,205,205))
  if flip:
    augmented = cv2.flip(augmented, 1)

  poster[25:75, 15:65] = augmented
  return poster

import cv2
import torch
import numpy as np
import os
import deepdish as dd
import json


poster_masks = {}
posters_alone = {}

poster_images = os.listdir("/content/drive/MyDrive/VLR_PROJECT/posters/augmented-samples-reduced")
poster_ind = 0

poster_locs = json.load(open("/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/wall_loc.json", "r"))

aug_mode = "poster_flip"
print("AUG MODE: " + aug_mode)

## Create the augmented posters ##
for k, v in surface_normals.items():

  room_name = k.split("_")[0]
  img_name = k[len(room_name)+1:]

  try:

    print("DOING " + room_name + "_" + img_name)

    test_room = '/content/drive/MyDrive/VLR_PROJECT/indoorCVPR_09/Images/' + room_name + "/" + img_name + '.jpg'
    test_depth = '/content/drive/MyDrive/VLR_PROJECT/depth_maps/depth_map_filtered/' + k + '.jpg'

    original_room = load_image(test_room)
    normal = v

    #Loc (x,y) in dictionary, but needs to be (y,x) here
    wall_loc = poster_locs[room_name + "/" + img_name + ".jpg"]
    loc = (wall_loc[1], wall_loc[0])
    poster_size = 40

    poster = load_image("/content/drive/MyDrive/VLR_PROJECT/posters/augmented-samples-reduced/" + poster_images[poster_ind])

    #Perform augmentation
    if aug_mode == "poster_rot":
      poster = augmentPoster(np.random.randint(-15, 15), 0)
    elif aug_mode == "poster_flip":
      poster = augmentPoster(0, np.random.random() > 0.5)
    else:
      print("INVALID AUG MODE")
      assert(1==0)

    new_room, poster_warped, poster_mask, poster_alone = poster_warp(poster, normal, loc, poster_size, np.copy(original_room))
    cv2.imwrite(os.path.join("/content/drive/MyDrive/VLR_PROJECT/test_poster_v4/" + aug_mode + "/poster" + str(poster_ind), k + "_room.png"), original_room)
    cv2.imwrite(os.path.join("/content/drive/MyDrive/VLR_PROJECT/test_poster_v4/" + aug_mode + "/poster" + str(poster_ind), k + "_normal.png"), normal_images[k])
    cv2.imwrite(os.path.join("/content/drive/MyDrive/VLR_PROJECT/test_poster_v4/" + aug_mode + "/poster" + str(poster_ind), k + "_test_result.png"), new_room)
    cv2.imwrite(os.path.join("/content/drive/MyDrive/VLR_PROJECT/test_poster_v4/" + aug_mode + "/poster" + str(poster_ind), k + "_warped_poster.png"), poster_warped)
    cv2.imwrite(os.path.join("/content/drive/MyDrive/VLR_PROJECT/test_poster_v4/" + aug_mode + "/poster" + str(poster_ind), k + "_poster_alone.png"), poster_alone)
    print(f"Image{k} is done")
    poster_masks[k] = poster_mask
    posters_alone[k] = poster_alone

    poster_ind = (poster_ind + 1) % len(poster_images)
  except:
    print("FAILED ON " + room_name + "_" + img_name)

dd.io.save("/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/warped_poster_masks" + aug_mode + ".h5", poster_masks)
dd.io.save("/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/warped_posters_alone" + aug_mode + ".h5", posters_alone)

###
import cv2
import torch
import numpy as np
import os
import deepdish as dd
import json


print("NOT WARPING")


poster_images = os.listdir("/content/drive/MyDrive/VLR_PROJECT/posters/augmented-samples-reduced")

poster_locs = json.load(open("/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/wall_loc.json", "r"))

aug_mode = "poster_rot"
print("AUG MODE: " + aug_mode)

#Created unwarped posters
for k, v in surface_normals.items():

  room_name = k.split("_")[0]
  img_name = k[len(room_name)+1:]

  for poster_ind in range(len(poster_images)):

    try:

      print("DOING " + room_name + "_" + img_name)

      test_room = '/content/drive/MyDrive/VLR_PROJECT/indoorCVPR_09/Images/' + room_name + "/" + img_name + '.jpg'
      test_depth = '/content/drive/MyDrive/VLR_PROJECT/depth_maps/depth_map_filtered/' + k + '.jpg'

      original_room = load_image(test_room)
      normal = v

      #Loc (x,y) in dictionary, but needs to be (y,x) here
      wall_loc = poster_locs[room_name + "/" + img_name + ".jpg"]
      loc = (wall_loc[1], wall_loc[0])
      poster_size = 40

      poster = load_image("/content/drive/MyDrive/VLR_PROJECT/posters/augmented-samples-reduced/" + poster_images[poster_ind])

      #Perform augmentation
      if aug_mode == "poster_rot":
        poster = augmentPoster(np.random.randint(-15, 15), 0)
      elif aug_mode == "poster_flip":
        poster = augmentPoster(0, np.random.random() > 0.5)
      else:
        print("INVALID AUG MODE")
        assert(1==0)

      new_room, poster_warped, poster_mask, poster_alone = no_poster_warp(poster, normal, loc, poster_size, np.copy(original_room))
      cv2.imwrite(os.path.join("/content/drive/MyDrive/VLR_PROJECT/test_poster_v4_nowarp/" + aug_mode + "/poster" + str(poster_ind), k + "_room.png"), original_room)
      cv2.imwrite(os.path.join("/content/drive/MyDrive/VLR_PROJECT/test_poster_v4_nowarp/" + aug_mode + "/poster" + str(poster_ind), k + "_test_result.png"), new_room)
      print(f"Image{k} is done")

    except:
      print("FAILED ON " + room_name + "_" + img_name)

"""


## Loading the stable diffusion model"""

#This only works for colab
from huggingface_hub import notebook_login
notebook_login()

import PIL
from diffusers import StableDiffusionInpaintPipeline

device = "cuda"
model_path = "runwayml/stable-diffusion-inpainting"

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
).to(device)


#Apply diffusion
def addImageDiffusion(img, mask):
  #prompt = "a " + room_type + " with a poster"
  #prompt = "a wall with a poster on it"
  prompt = "the edge of a black frame blended into the wall in the background"

  results = pipe(
      prompt=prompt,
      image=img,
      mask_image=img_mask,
      guidance_scale=7.0,  #How faithful to prompt
      generator=torch.Generator(device="cuda").manual_seed(0),
      num_images_per_prompt=1,
  ).images

  return results[0]



from PIL import Image
import numpy as np
import deepdish as dd
import os
import cv2


aug_mode = "poster_flip"
print("AUG MODE: " + aug_mode)

poster_masks = dd.io.load("/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/warped_poster_masks" + aug_mode + ".h5")


def get_poster_ind(name):
  for i in range(7):
    path = '/content/drive/MyDrive/VLR_PROJECT/test_poster_v4/' + aug_mode + "/poster" + str(i) + "/" + name + '_test_result.png'
    if os.path.exists(path):
      return i
  print("FAILED TO FIND POSTER INDEX")
  assert(1==0)

import time

start = time.time()

#Blend the poster to background with stable diffusion image
for k, img_mask in poster_masks.items():

  poster_ind = get_poster_ind(k)

  test_room = '/content/drive/MyDrive/VLR_PROJECT/test_poster_v4/' + aug_mode + "/poster" + str(poster_ind) + "/" + k + '_test_result.png'

  img = Image.open(test_room)

  #Create mask for diffusing the area around the poster
  kernel = np.ones((5, 5), np.uint8)
  img_mask_dilate = cv2.dilate(img_mask, kernel, iterations=2)
  img_mask_dilate -= img_mask
  img_mask = np.copy(img_mask_dilate * 255)

  #Formatting
  width, height = img.size
  img = img.resize((512, 512)).convert('RGB')
  img_mask = Image.fromarray(img_mask)
  img_mask = img_mask.resize((512, 512))


  #Applying the diffusion
  result = addImageDiffusion(img, img_mask)
  result = result.resize((width, height))

  blended = Image.open(test_room) * (1-img_mask_dilate) + result * img_mask_dilate
  blended = Image.fromarray(blended)

  img_mask = img_mask.resize((width, height))
  img_mask.save(os.path.join("/content/drive/MyDrive/VLR_PROJECT/blended_images/" + aug_mode + "/poster" + str(poster_ind), k + "_mask_dilated.png"))
  blended.save(os.path.join("/content/drive/MyDrive/VLR_PROJECT/blended_images/" + aug_mode + "/poster" + str(poster_ind), k + "_test_result.png"))

print(time.time()-start)

"""## Formatting dataset"""

import random
import shutil
import os

#Merge datasets and do horizontal flipping
import cv2

def flip_room(path):
  img = cv2.imread(path)
  flipped = cv2.flip(img, 1)
  return flipped

no_warped = "/content/drive/MyDrive/VLR_PROJECT/test_poster_v4_nowarp/"
warped = "/content/drive/MyDrive/VLR_PROJECT/test_poster_v4/"
blended = "/content/drive/MyDrive/VLR_PROJECT/blended_images/"

task3_folder = "/content/drive/MyDrive/VLR_PROJECT/Task3_Images"

paths = []


for method in ["no_warp", "warp", "warp_diffuse"]:
  for aug in ["poster_flip", "poster_rot"]:
    for room_flip in ["normal", "flipped"]:
      os.makedirs(os.path.join(task3_folder, method, aug, room_flip))


#Moving data
for poster_ind in range(7):
  for aug in ["poster_flip", "poster_rot"]:
    images = os.listdir(blended + aug + "/poster" + str(poster_ind))

    images = filter(lambda k: 'mask_dilated' not in k, images)
    images = [i[:-16] + ".png" for i in images]
    print(images)

    random.shuffle(images)
    images = images[0:15]

    for im in images:
      im_test_result = im[:-4] + "_test_result" + ".png"
      im_save = "_".join(im.split("_")[1:])

      print(im)
      print(im_save)

      #Unwarped
      source = no_warped + aug + "/poster" + str(poster_ind) + "/" + im_test_result
      destination = os.path.join(task3_folder, "no_warp", aug, "normal", im_save)
      paths.append(destination)
      shutil.copy(source, destination)

      flipped_nowarp = flip_room(source)
      destination = os.path.join(task3_folder, "no_warp", aug, "flipped", im_save)
      paths.append(destination)
      cv2.imwrite(destination, flipped_nowarp)


      #Warped
      source = warped + aug + "/poster" + str(poster_ind) + "/" + im_test_result
      destination = os.path.join(task3_folder, "warp", aug, "normal", im_save)
      paths.append(destination)
      shutil.copy(source, destination)

      flipped_warp = flip_room(source)
      destination = os.path.join(task3_folder, "warp", aug, "flipped", im_save)
      paths.append(destination)
      cv2.imwrite(destination, flipped_warp)


      #Warped and diffused
      source = blended + aug + "/poster" + str(poster_ind) + "/" + im_test_result
      destination = os.path.join(task3_folder, "warp_diffuse", aug, "normal", im_save)
      paths.append(destination)
      shutil.copy(source, destination)

      flipped_warpdiffuse = flip_room(source)
      destination = os.path.join(task3_folder, "warp_diffuse", aug, "flipped", im_save)
      paths.append(destination)
      cv2.imwrite(destination, flipped_warpdiffuse)

import deepdish as dd
poster_masks_flip = dd.io.load("/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/warped_poster_masks" + "poster_flip" + ".h5")
poster_masks_rot = dd.io.load("/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/warped_poster_masks" + "poster_rot" + ".h5")

import pandas as pd
import os
import glob
import cv2

#Get the corners of the warped poster
def get_warped_corners_warped(name, aug_mode):
  masks = None
  if aug_mode == "poster_rot":
    masks = poster_masks_rot
  elif aug_mode == "poster_flip":
    masks = poster_masks_flip

  mask = masks[name]

  ys, xs, _ = np.asarray(np.nonzero(mask))

  min_x = np.min(xs)
  min_y = np.min(ys)

  max_x = np.max(xs)
  max_y = np.max(ys)

  return (min_x, min_y), (max_x, max_y)

#Get the corners of the unwarped poster
def get_unwarped_corners(name, aug_mode, path):
  parent = "/".join(path.split("/")[:-1])
  no_poster = cv2.imread(parent + "/" + name + "_room.png")
  with_poster = cv2.imread(parent + "/" + name + "_test_result.png")
  mask = (no_poster - with_poster) != 0

  ys, xs, _ = np.asarray(np.nonzero(mask))

  min_x = np.min(xs)
  min_y = np.min(ys)

  max_x = np.max(xs)
  max_y = np.max(ys)

  return (min_x, min_y), (max_x, max_y)


#Get poster corners
def get_corners(name, aug_mode, method, path):
  if method == "no_warp":
    return get_unwarped_corners(name, aug_mode, path)
  else:
    return get_warped_corners_warped(name, aug_mode)

unwarp_count = 0


#Get metadata for each image
def get_image_info(path, method):
  values = path.split("/")
  mode = values[6]
  aug_mode = values[7]
  room_mode = values[8]
  file_name = values[9][:-4]
  parent = None
  if method == "no_warp":
    parent = '/content/drive/MyDrive/VLR_PROJECT/test_poster_v4_nowarp/'
  elif method == "warp":
    parent = '/content/drive/MyDrive/VLR_PROJECT/test_poster_v4/'
  elif method == "warp_diffuse":
    parent = '/content/drive/MyDrive/VLR_PROJECT/blended_images/'
  else:
    print("ERROR: INVALID METHOD")
    assert(1==0)
  for i in range(7):

    global unwarp_count
    if method == "no_warp":
      i = unwarp_count
      unwarp_count += 1
      if unwarp_count >= 7:
        unwarp_count = 0

    path_incomplete = parent + aug_mode + "/poster" + str(i) + "/*" + file_name + '_test_result.png'
    possible_files = glob.glob(path_incomplete)
    modified = []
    rooms = []
    for f in possible_files:
      name = f.split("/")[8]
      room = name.split("_")[0]
      name_noRoom = "_".join(name.split("_")[1:])
      comb_list = f.split("/")[0:8] + [name_noRoom]
      new_f = "/".join(comb_list)
      if room + "_" + file_name in f:
        modified.append(f)
        rooms.append(room)
    if not (len(modified) <= 1):

      assert("salon19" in modified[0])
      (min_x, min_y), (max_x, max_y) = get_corners(name[:-16], aug_mode, method, modified[0])
      return i, modified[0], rooms[0], (min_x, min_y), (max_x, max_y)#.replace("salon19", "b_salon19"), rooms[0]
    if len(modified) == 1:
      if method == "no_warp":
        print(modified[0])
      (min_x, min_y), (max_x, max_y) = get_corners(name[:-16], aug_mode, method, modified[0])
      return i, modified[0], rooms[0], (min_x, min_y), (max_x, max_y)
  #print("FAILED TO FIND POSTER INDEX")
  #print("INFO")
  #print(path)
  assert(1==0)


results = []
objs = ["apple", "bowl", "camera", "cellphone", "cup", "orange", "water-bottle"]

for check_path, directories, files in os.walk('/content/drive/MyDrive/VLR_PROJECT/Task3_Images/'):
  for file in files:
    method = check_path.split("/")[6]
    poster_id, path, room, (min_x, min_y), (max_x, max_y) = get_image_info(check_path + "/" + file, method)
    results.append([path, min_x, min_y, max_x, max_y, objs[poster_id], room, method])

df = pd.DataFrame(results)
df.to_csv(f"/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/Task3_info_unformatted.csv")

"""


## Renaming and moving files
"""

#Reformat data
import csv

import shutil


task3_folder_new = "/content/drive/MyDrive/VLR_PROJECT/Task3_Images_Formatted"
new_results = []

#Save to CSV
count = 0
with open(f"/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/Task3_info_unformatted.csv", mode ='r') as file:
  csvFile = csv.reader(file)
  for lines in csvFile:
        if len(lines[0]) > 0:
          print(lines)
          count += 1
          path = lines[1]
          room = lines[7]
          obj = lines[6]
          method = lines[8]
          print(method, obj)
          file_name = path.split("/")[-1]
          parent_folder = task3_folder_new + "/" + method + "/poster-" + obj + "/" + room
          destination = parent_folder + "/" + file_name
          if not os.path.exists(parent_folder):
            os.makedirs(parent_folder)
          shutil.copy(path, destination)
          new_results.append([destination, lines[2], lines[3], lines[4], lines[5], lines[6], lines[7], lines[8]])

assert(count == 420*3)
df = pd.DataFrame(new_results)
df.to_csv(f"/content/drive/MyDrive/VLR_PROJECT/Dictionaries_and_lists/Task3_info.csv")

#'/content/drive/MyDrive/VLR_PROJECT/test_poster_v4/poster_flip/poster0/classroom_classroom1176_test_result.png'
#classroom_aulas_007_room.png
import os
import shutil
import csv


original_folder = '/content/drive/MyDrive/VLR_PROJECT/Task3_OriginalImages/'
shutil.rmtree(original_folder)
os.makedirs(original_folder)

#Move data
original_paths = []
for root,dirs,files in os.walk('/content/drive/MyDrive/VLR_PROJECT/test_poster_v4'):
  for f in files:
    if f[-8:-4] == "room":
      parsed = f.split("_")
      roomType = parsed[0]
      removed_roomType = "_".join(parsed[1:])
      if not os.path.exists(original_folder + roomType):
        os.makedirs(original_folder + roomType)
      shutil.copy(root + "/" + f, original_folder + roomType + "/" + removed_roomType[:-9] + ".png")