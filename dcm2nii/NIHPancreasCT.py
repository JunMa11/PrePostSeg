# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:18:05 2019
Convert following pancreas CT dcm dataset to nii
https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT
@author: Jun
"""

import os
import SimpleITK as sitk
import nibabel as nb
import numpy as np
import re
import glob

path = r'Pancreas-CT' # dcm path
save_path = r'Pancreas\NIH'
filenames = os.listdir(path)
filenames.sort()

for name in filenames:
    namepath = os.path.join(path, name)
    dicom_names = glob.glob(namepath + r'\*\*\*.dcm')
    dicom_names.sort()
    DicomReader = sitk.ImageSeriesReader()
#    seriesIDs = DicomReader.GetGDCMSeriesIDs(filepath)
    
#    dicom_names = DicomReader.GetGDCMSeriesFileNames(filepath, seriesIDs[0])
    DicomReader.SetFileNames(dicom_names)
    LiverImg = DicomReader.Execute()
#    LiverData = sitk.GetArrayFromImage(LiverImg)
    save_name = name + '.nii.gz'
    sitk.WriteImage(LiverImg, os.path.join(save_path, save_name))   
    print(save_name)



