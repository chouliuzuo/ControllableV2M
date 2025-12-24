import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
# import sys
# sys.path.append("..")
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        print(m)
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
        

# image = cv2.imread('2.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# sam_checkpoint = "sam_vit_h_4b8939.pth"
# model_type = "vit_h"
# device = "cuda:0"
# sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)
# mask_generator = SamAutomaticMaskGenerator(sam)
# masks = mask_generator.generate(image)
# plt.figure(figsize=(20,20))
# plt.imshow(image)
# show_anns(masks)
# plt.axis('off')
# plt.show() 
# plt.savefig('testkk.jpg')

import numpy as np
 
# 创建一个2D矩阵
matrix = np.array([[1, 2], [3, 4]])
# 复制矩阵到3D体积，增加一个维度
new_matrix = np.concatenate([matrix,matrix,matrix],axis=0)
new_matrix = new_matrix.reshape((3,2,2)).transpose(1,2,0)
 
print(new_matrix)
print(new_matrix.shape)