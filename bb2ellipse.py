# -*- coding: utf-8 -*-
"""
Created on Tue May  7 11:25:53 2019

@author: JUN
"""

import numpy as np
import os
import nibabel as nb
from skimage import measure

file = r'\Downloads\annotations_of_training_set\segmentation-1.txt'

with open(file, "r") as f:
    bboxes = f.read()

bboxes = bboxes.splitlines()
label_num = len(bboxes)

liver_path = r'LITS\imagesTr'
save_path =  r'LITS'

img_nii = nb.load(os.path.join(liver_path, 'train_1_0000.nii.gz'))
img_data = img_nii.get_data()
img_label = np.zeros_like(img_data, dtype=int)

for i in range(0, label_num):
    bb_info = bboxes[i].split(' ')
    label = int(bb_info[1])
    xmin = int(bb_info[2])
    xmax = int(bb_info[3])
    ymin = int(bb_info[4])
    ymax = int(bb_info[5])
    zmin = int(bb_info[6])
    zmax = int(bb_info[7])
    img_label_temp = np.zeros_like(img_data, dtype=int)
    img_label_temp[xmin:xmax, ymin:ymax, zmin:zmax] = 1
    props = measure.regionprops(img_label_temp)
    bb_centroid = np.array(props[0]['centroid'], dtype=int)
    r_x = (xmax-xmin)/2; r_y = (ymax-ymin)/2; r_z = (zmax-zmin)/2
    xlim, ylim, zlim = np.ogrid[0:float(img_data.shape[0]), 0:float(img_data.shape[1]), 0:float(img_data.shape[2])]
    distance = ((xlim-bb_centroid[0])/r_x)**2 + ((ylim-bb_centroid[1])/r_y)**2 + ((zlim-bb_centroid[2])/r_z)**2
    img_label[distance<1] = label 
    
save_nii = nb.Nifti1Image(img_label, img_nii.affine, img_nii.header)
nb.save(save_nii, os.path.join(save_path, 'bb_ellipse.nii.gz'))

