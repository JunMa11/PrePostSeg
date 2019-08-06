# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 13:45:29 2019
Make CHAO training data
@author: Administrator
"""


import nibabel as nb
import os
import re
#import SimpleITK as sitk
#import skimage.measure as sm
import numpy as np
#import matplotlib.pyplopip t as plt
#from sklearn.decomposition import FastICA, PCA
#import scipy
import shutil


#%% train data CT
img_path = r'\Train_Sets\CT_nii'
save_path = r'\LiverCT\imagesTr'

filenames = os.listdir(img_path)
filenames.sort()

def cutoff_img(img, lower=-2000, upper=2000):
    img_up = np.where(img>upper, upper, img)
    img_lower = np.where(img_up<lower, lower, img_up)
    
    return img_lower

for name in filenames:
    num = re.findall('\d+', name)[0]
    img_nii = nb.load(os.path.join(img_path, name+'\\CT_image.nii.gz'))
    img_data = img_nii.get_data()
    img_data = cutoff_img(img_data)
    newimg_nii = nb.Nifti1Image(img_data, img_nii.affine, img_nii.header)    
    new_img_name = 'liverct_' + num + '_0000.nii.gz'
    label_data_path = os.path.join(img_path, name+'\\liver_mask.nii.gz')
    new_label_name = 'liverct_' + num + '.nii.gz'

    nb.save(newimg_nii, os.path.join(save_path, new_img_name))
    shutil.copyfile(label_data_path, os.path.join(save_path, new_label_name))

    
    
#%% test data ct    
img_path = r'\Test_Sets\CT_nii'
save_path = r'\LiverCT\imagesTs'

filenames = os.listdir(img_path)
filenames.sort()

def cutoff_img(img, lower=-2000, upper=2000):
    img_up = np.where(img>upper, upper, img)
    img_lower = np.where(img_up<lower, lower, img_up)
    
    return img_lower

for name in filenames:
    num = re.findall('\d+', name)[0]
    img_nii = nb.load(os.path.join(img_path, name+'\\CT_image.nii.gz'))
    img_data = img_nii.get_data()
    img_data = cutoff_img(img_data)
    newimg_nii = nb.Nifti1Image(img_data, img_nii.affine, img_nii.header)    
    new_img_name = 'liverct_' + num + '_0000.nii.gz'

    nb.save(newimg_nii, os.path.join(save_path, new_img_name)) 
    
    
#%% train data MR
img_path = r'\Train_Sets\MR_nii'
save_path = r'MR\imagesTr'

filenames = os.listdir(img_path)
filenames.sort()

for name in filenames:
    num = re.findall('\d+', name)[0]
    # t1 in phase
    t1in_path = os.path.join(img_path, name+'\\T1DUAL_in_phase_image.nii.gz')
    new_t1in = os.path.join(save_path, 't1in_' + num + '_0000.nii.gz')
    shutil.copyfile(t1in_path, new_t1in)
    # t1 out phase
    t1out_path = os.path.join(img_path, name+'\\T1DUAL_out_phase_image.nii.gz')
    new_t1out = os.path.join(save_path, 't1out_' + num + '_0000.nii.gz')
    shutil.copyfile(t1out_path, new_t1out)
    # t1 gt
    t1gt_path = os.path.join(img_path, name+'\\T1DUAL_mask.nii.gz')
    shutil.copyfile(t1gt_path, os.path.join(save_path, 't1in_' + num + '.nii.gz'))
    shutil.copyfile(t1gt_path, os.path.join(save_path, 't1out_' + num + '.nii.gz'))

    # t2 
    t2_path = os.path.join(img_path, name+'\\T2SPIR_image.nii.gz')
    new_t2 = os.path.join(save_path, 't2_' + num + '_0000.nii.gz')
    shutil.copyfile(t2_path, new_t2)
    # t1 gt
    t2gt_path = os.path.join(img_path, name+'\\T2SPIR_mask.nii.gz')
    shutil.copyfile(t2gt_path, os.path.join(save_path, 't2_' + num + '.nii.gz'))

    
#%% test data mr
img_path = r'\Test_Sets\MR_nii'
save_path = r'MR\imagesTs'

filenames = os.listdir(img_path)
filenames.sort()

for name in filenames:
    num = re.findall('\d+', name)[0]
    # t1 in phase
    t1in_path = os.path.join(img_path, name+'\\T1DUAL_in_phase_image.nii.gz')
    new_t1in = os.path.join(save_path, 't1in_' + num + '_0000.nii.gz')
    shutil.copyfile(t1in_path, new_t1in)
    # t1 out phase
    t1out_path = os.path.join(img_path, name+'\\T1DUAL_out_phase_image.nii.gz')
    new_t1out = os.path.join(save_path, 't1out_' + num + '_0000.nii.gz')
    shutil.copyfile(t1out_path, new_t1out)

    # t2 
    t2_path = os.path.join(img_path, name+'\\T2SPIR_image.nii.gz')
    new_t2 = os.path.join(save_path, 't2_' + num + '_0000.nii.gz')
    shutil.copyfile(t2_path, new_t2)
   
    

#%% extract liver label from MR multi-organ label
label_path = r'\MR\labelsTr'
save_path = r'LiverMR\labelsTr'

filenames = os.listdir(label_path)
filenames.sort()

for name in filenames:
    gt_nii = nb.load(os.path.join(label_path, name))
    gt_data = gt_nii.get_data()
    gt_data[gt_data>1] = 0
    
    save_nii = nb.Nifti1Image(gt_data, gt_nii.affine, gt_nii.header)
    nb.save(save_nii, os.path.join(save_path, name))











    
    