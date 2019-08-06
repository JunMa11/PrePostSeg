# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:06:18 2019
Make bounding box according to ground truth
@author: Jun
"""


import nibabel as nb
import numpy as np
import os, re
join = os.path.join
from medpy import metric
from collections import OrderedDict
import pandas as pd
#from skimage import measure

#%%

def find_bb(volume):
    img_shape = volume.shape
    bb = np.zeros((6,), dtype=np.uint)
    bb_extend = 10
    # axis
    for i in range(img_shape[0]):
        img_slice_begin = volume[i,:,:]
        if np.sum(img_slice_begin)>0:
            bb[0] = np.max([i-bb_extend, 0])
            break

    for i in range(img_shape[0]):
        img_slice_end = volume[img_shape[0]-1-i,:,:]
        if np.sum(img_slice_end)>0:
            bb[1] = np.min([img_shape[0]-1-i + bb_extend, img_shape[0]-1])
            break
    # seg
    for i in range(img_shape[1]):
        img_slice_begin = volume[:,i,:]
        if np.sum(img_slice_begin)>0:
            bb[2] = np.max([i-bb_extend, 0])
            break

    for i in range(img_shape[1]):
        img_slice_end = volume[:,img_shape[1]-1-i,:]
        if np.sum(img_slice_end)>0:
            bb[3] = np.min([img_shape[1]-1-i + bb_extend, img_shape[1]-1])
            break

    # coronal
    for i in range(img_shape[2]):
        img_slice_begin = volume[:,:,i]
        if np.sum(img_slice_begin)>0:
            bb[4] = np.max([i-bb_extend, 0])
            break

    for i in range(img_shape[2]):
        img_slice_end = volume[:,:,img_shape[2]-1-i]
        if np.sum(img_slice_end)>0:
            bb[5] = np.min([img_shape[2]-1-i+bb_extend, img_shape[2]-1])
            break
    
    return bb

#%%
    
img_path = r'KITS\imagesTs'
#gt_path = r'\labelsTr'
seg_path = r'KITS\Kidney'
save_path = r'KITS\KidRoi'

names = os.listdir(seg_path)
names.sort()

records = OrderedDict()
records['name'] = list()
records['bb'] = list()

for name in names:
    records['name'].append(name)
    name_prefix = name.split('.nii.gz')[0]
    img_nii = nb.load(join(img_path, name_prefix + '_0000.nii.gz'))
    img_data = img_nii.get_data()
#    gt_data = nb.load(join(gt_path, name)).get_data()
    seg_data = nb.load(join(seg_path, name)).get_data()
    
#    recall = metric.recall(seg_data, gt_data>0)
#    print(name, recall)
#    if recall < 0.96:
#        bb = find_bb(gt_data)
#    else:
    bb = find_bb(seg_data)
    records['bb'].append(bb)
     
    img_roi = img_data[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
#    seg_roi = seg_data[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]]
#    check_seg = np.zeros_like(gt_data)
#    check_seg[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]] = 1
#    if metric.recall(check_seg, gt_data>0) < 0.99:
#        print(name + 'recall: ', metric.recall(check_seg, gt_data>0))
    
    volume_roi_nii = nb.Nifti1Image(img_roi, img_nii.affine, img_nii.header)
#    gt_roi_nii = nb.Nifti1Image(seg_roi, img_nii.affine, img_nii.header)
    
    nb.save(volume_roi_nii, join(save_path, name_prefix + '_0000.nii.gz'))
#    nb.save(gt_roi_nii, join(save_path,  name))




dataframe = pd.DataFrame(records)
dataframe.to_csv(r'KITS\Roibb.csv', index=False)


#%% put seg into original image
roipath = r'KITS\Roi3DTopK'
kidneypath = r'KITS\Kidney'
save_path = r'KITS\Roi3DTopKFinal'
#bb_df = pd.read_csv(r'KITS\Roibb.csv')
names = os.listdir(roipath)
names.sort()

for name in names:
    roi_data = nb.load(join(roipath, name)).get_data()
    kid_nii = nb.load(join(kidneypath, name))
    kid_data = kid_nii.get_data()
    bb = find_bb(kid_data)
    final_seg = np.zeros_like(kid_data, dtype=np.uint8)
    final_seg[bb[0]:bb[1], bb[2]:bb[3], bb[4]:bb[5]] = roi_data
    final_nii = nb.Nifti1Image(final_seg, kid_nii.affine, kid_nii.header)
    nb.save(final_nii, join(save_path, name))
    
    




