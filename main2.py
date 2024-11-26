from PIL import Image
import numpy as np
from sklearn.metrics import jaccard_score
import os
import torch
import imageio.v2 as imageio
os.environ['TORCH'] = torch.__version__
#print(torch.__version__)

folder1 = "C:/Users/tm988/Desktop/MajorProj/codes/UnSeGNet/CVC-ClinicDB_2/CVC-ClinicDB/Original"
folder2 = "C:/Users/tm988/Desktop/MajorProj/codes/UnSeGNet/CVC-ClinicDB_2/CVC-ClinicDB/Ground Truth"

images = sorted(os.listdir(folder1))
masks = sorted(os.listdir(folder2))

images = [folder1 + '/' + x for x in images]
masks = [folder2 + '/' + x for x in masks]

images = images[1:-1]
masks = masks[1:-1]


for imagep, true_maskp in zip(images, masks):
    #img = np.asarray(Image.open(imagep))
    img = imageio.imread(imagep)
    seg = np.asarray(Image.open(true_maskp).convert('L'))
    true_mask = np.where(seg == 255,1,0).astype(np.uint8)

# print(len(masks), len(images)) # 610 610

# print(Image.open(masks[22]).size) #(384, 288)

#print(np.max(np.asarray(Image.open(masks[22]).convert('L'))) ) # 255

"""
mask = masks[22]
print(mask) # C:/Users/tm988/Desktop/MajorProj/codes/UnSeGNet/CVC-ClinicDB_2/CVC-ClinicDB/Ground Truth/12.tif
print("---------------------")
seg = np.asarray(Image.open(mask).convert('L'))
print(seg) # [[0 0 0 ... 0 0 0]...[0 0 0 ... 0 0 0]]
print("---------------------")
true_mask = np.where(seg == 255,1,0).astype(np.uint8)
print(true_mask) # [[0 0 0 ... 0 0 0]...[0 0 0 ... 0 0 0]]
print("---------------------")
"""

def iou(mask1, mask2):
    # mask1 = mask1.ravel()
    # mask2 = mask2.ravel()
    # iou = jaccard_score(mask1, mask2)
    # return iou

    x = mask1.ravel()
    y = mask2.ravel()
    intersection = np.logical_and(x, y)
    union = np.logical_or(x, y)
    similarity = np.sum(intersection)/ np.sum(union)
    return similarity

def cts(image, segmap,wnd = [20,20]):
    """
    Crop the image and segmentation map to the boundary of the segmentation.

    :param image: A numpy array representing the color image.
    :param segmap: A numpy array representing the binary segmentation map.
    :return: Cropped image and segmentation map.
    """
    [ht,wdt] = segmap.shape
    # Find the indices where segmap is 1
    rows, cols = np.where(segmap == 255)
    # Find min and max coordinates
    min_row, max_row = max(min(rows)-wnd[0],0), min(max(rows)+wnd[0],ht)
    min_col, max_col = max(min(cols)-wnd[1],0), min(max(cols)+wnd[1],wdt)

    # Crop the image and segmap
    cropped_image = image[min_row:max_row+1, min_col:max_col+1]
    cropped_segmap = segmap[min_row:max_row+1, min_col:max_col+1]

    return cropped_image, cropped_segmap

"""
img = imageio.imread(images[22])
seg = np.asarray(Image.open(masks[22]).convert('L'))

img, seg = cts(img, seg)
"""