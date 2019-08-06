# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:45:38 2019

@author: Jun
"""

import os, re
import nibabel as nb 
from skimage import morphology, measure, io
import numpy as np
import scipy
import SimpleITK as sitk

join = os.path.join

def getLargestCC(segmentation):
    labels = measure.label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC




# <teamName>\<Task#>\CT\<setnumber>\Results\img000.png
#%% Task2 Liver segmentation in CT
pred_path = r'\InferenceResults\LiverCT\2D'
save_path = r'\FightHCC2D\Task2\CT'

filenames = os.listdir(save_path)
filenames.sort()

for name in filenames:
    folder_path = join(save_path, name)
    nii_name = 'liverct_' + name + '.nii.gz'

    nii =  sitk.ReadImage(join(pred_path, nii_name))
    nii_data = sitk.GetArrayFromImage(nii)

    liver = np.uint8(getLargestCC(nii_data>0))*255
    slice_num = liver.shape[0]
    for i in range(liver.shape[0]):
        save_name = 'Results\\img' + ("%03d" % i) + '.png'
        io.imsave(join(folder_path, save_name), liver[slice_num-i-1,:,:])


#%% Task 3: liver segmentation in MR

save_path = r'\FightHCC3DV2\Task3\MR'
pred_path = r'\InferenceResults\LiverMR\TrSeg'

filenames = os.listdir(save_path)
filenames.sort()

# T1DUAL
for name in filenames:
    folder_path = join(save_path, name)
    nii_name = 't1in_' + name + '.nii.gz'

#   nii_data = nb.load(join(pred_path, nii_name)).get_data()
    nii =  sitk.ReadImage(join(pred_path, nii_name))
    nii_data = sitk.GetArrayFromImage(nii)
    liver = getLargestCC(nii_data>0)
    liver = morphology.remove_small_holes(liver, 100000)
    liver = np.uint8(liver)*63
    
    for i in range(liver.shape[0]):
        save_name = 'T1DUAL\\Results\\img' + ("%03d" % i) + '.png'
        io.imsave(join(folder_path, save_name), liver[i,:,:])



# Save T2SPIR_nii.gz to png
for name in filenames:
    folder_path = join(save_path, name)
    nii_name = 't2_' + name + '.nii.gz'
#    if os.path.exists(join(pred_path, nii_name)):
    nii =  sitk.ReadImage(join(pred_path, nii_name))
    nii_data = sitk.GetArrayFromImage(nii)
    liver = np.uint8(getLargestCC(nii_data>0))*63
    
    for i in range(liver.shape[0]):
        save_name = 'T2SPIR\\Results\\img' + ("%03d" % i) + '.png'
        io.imsave(join(folder_path, save_name), liver[i,:,:])





#%% Task25/24 MR Multi Organ
    
save_path = r'\FightHCC3D\Task4\MR'
pred_path = r'\InferenceResults\CTMR\3D'


filenames = os.listdir(save_path)
filenames.sort()

for name in filenames:
    folder_path = join(save_path, name)
    nii_name = 't1in_' + name + '.nii.gz'

    nii =  sitk.ReadImage(join(pred_path, nii_name))
    nii_data = sitk.GetArrayFromImage(nii)
    
    liver = getLargestCC(nii_data==1)
    liver = morphology.remove_small_holes(liver, 100000)
    liver = np.uint8(liver)*63
    
    right_kidney = np.uint8(getLargestCC(nii_data==2))*126
    left_kidney = np.uint8(getLargestCC(nii_data==3))*189
    spleen = np.uint8(getLargestCC(nii_data==4))*252
    
    final_seg = liver + right_kidney + left_kidney + spleen  
        
    for i in range(final_seg.shape[0]):
        save_name = 'T1DUAL\\Results\\img' + ("%03d" % i) + '.png'
        io.imsave(join(folder_path, save_name), final_seg[i,:,:])


for name in filenames:
    folder_path = join(save_path, name)
    nii_name = 't2_' + name + '.nii.gz'
#    if os.path.exists(join(pred_path, nii_name)):
    nii =  sitk.ReadImage(join(pred_path, nii_name))
    nii_data = sitk.GetArrayFromImage(nii)
    
    liver = getLargestCC(nii_data==1)
    liver = morphology.remove_small_holes(liver, 100000)
    liver = np.uint8(liver)*63
    
    right_kidney = np.uint8(getLargestCC(nii_data==2))*126
    left_kidney = np.uint8(getLargestCC(nii_data==3))*189
    spleen = np.uint8(getLargestCC(nii_data==4))*252
    
    final_seg = liver + right_kidney + left_kidney + spleen  
    
    for i in range(final_seg.shape[0]):
        save_name = 'T2SPIR\\Results\\img' + ("%03d" % i) + '.png'
        io.imsave(join(folder_path, save_name), final_seg[i,:,:])


#%% Task24 CT-Liver
save_path = r'\FightHCC3D\Task4\CT'
pred_path = r'\InferenceResults\CTMR\3D'

filenames = os.listdir(save_path)
filenames.sort()

for name in filenames:
    folder_path = join(save_path, name)
    nii_name = 'liverct_' + name + '.nii.gz'

    nii =  sitk.ReadImage(join(pred_path, nii_name))
    nii_data = sitk.GetArrayFromImage(nii)

    liver = np.uint8(getLargestCC(nii_data>0))*255
    slice_num = liver.shape[0]
    for i in range(liver.shape[0]):
        save_name = 'Results\\img' + ("%03d" % i) + '.png'
        io.imsave(join(folder_path, save_name), liver[slice_num-i-1,:,:])


#%% check MR shape
save_path = r'\FightHCC3DV2\Task3\MR'
dcm_path = r'\CHAOSdcm\Test_Sets\MR'

filenames = os.listdir(dcm_path)
filenames.sort()


for name in filenames:
    dcm_folder_path = join(dcm_path, name+'\\T1DUAL\\DICOM_anon\\InPhase')
    dcm_files = os.listdir(dcm_folder_path)
    
    png_folder = join(save_path, name+'\\T1DUAL\\Results')
    png_files = os.listdir(png_folder)
    
    assert len(dcm_files)==len(png_files), name + "image num mismatch"
    dcm_data = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(join(dcm_folder_path, dcm_files[0]))))
    dcm_data_shape = dcm_data.shape
    png_data = io.imread(join(png_folder, png_files[0]))
    assert dcm_data_shape == png_data.shape, name + "shape mismatch"
    print(name)



for name in filenames:
    dcm_folder_path = join(dcm_path, name+'\\T2SPIR\\DICOM_anon')
    dcm_files = os.listdir(dcm_folder_path)
    
    png_folder = join(save_path, name+'\\T2SPIR\\Results')
    png_files = os.listdir(png_folder)
    
    assert len(dcm_files)==len(png_files), name + "image num mismatch"
    dcm_data = np.squeeze(sitk.GetArrayFromImage(sitk.ReadImage(join(dcm_folder_path, dcm_files[0]))))
    dcm_data_shape = dcm_data.shape
    png_data = io.imread(join(png_folder, png_files[0]))
    assert dcm_data_shape == png_data.shape, name + "shape mismatch"


print('Finished, No error!')

