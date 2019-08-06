# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:14:39 2019

@author: Jun
"""

import os
import shutil
import nibabel as nb
import numpy as np
from collections import OrderedDict
import pandas as pd
join = os.path.join


#%%
hgg_path = r'\MICCAI_BraTS_2019_Data_Training\HGG'
lgg_path = r'\MICCAI_BraTS_2019_Data_Training\LGG'
save_imgpath = r'Brats\imagesTr'
save_gtpath = r'Brats\labelsTr'

hggs = os.listdir(hgg_path)
hggs.sort()
lggs = os.listdir(lgg_path)
lggs.sort()

i = 0
for hgg in hggs:
    hgg_casepath = join(hgg_path, hgg)
    names = os.listdir(hgg_casepath)
    names.sort()
    shutil.copyfile(join(hgg_casepath, names[0]), join(save_imgpath, 'brats_' + str(i) + '_0000.nii.gz')) # flair
    shutil.copyfile(join(hgg_casepath, names[1]), join(save_gtpath, 'brats_' + str(i) + '.nii.gz'))
    shutil.copyfile(join(hgg_casepath, names[2]), join(save_imgpath, 'brats_' + str(i) + '_0003.nii.gz')) # t1
    shutil.copyfile(join(hgg_casepath, names[3]), join(save_imgpath, 'brats_' + str(i) + '_0002.nii.gz')) # t1ce
    shutil.copyfile(join(hgg_casepath, names[4]), join(save_imgpath, 'brats_' + str(i) + '_0001.nii.gz')) # t2
    
    i+=1


for lgg in lggs:
    lgg_casepath = join(lgg_path, lgg)
    names = os.listdir(lgg_casepath)
    names.sort()
    shutil.copyfile(join(lgg_casepath, names[0]), join(save_imgpath, 'brats_' + str(i) + '_0000.nii.gz'))
    shutil.copyfile(join(lgg_casepath, names[1]), join(save_gtpath, 'brats_' + str(i) + '.nii.gz'))
    shutil.copyfile(join(lgg_casepath, names[2]), join(save_imgpath, 'brats_' + str(i) + '_0003.nii.gz'))
    shutil.copyfile(join(lgg_casepath, names[3]), join(save_imgpath, 'brats_' + str(i) + '_0002.nii.gz'))
    shutil.copyfile(join(lgg_casepath, names[4]), join(save_imgpath, 'brats_' + str(i) + '_0001.nii.gz'))
    i+=1

#%% rename validation and test data

valdata_path = r'BraTS\MICCAI_BraTS_2019_Data_Validation'
save_imgpath = r'BraTS\imagesTs'

cases = os.listdir(valdata_path)
cases.sort()

for case in cases:
    subfilepath = join(valdata_path, case)
    names = os.listdir(subfilepath)
    shutil.copyfile(join(subfilepath, names[0]), join(save_imgpath, case + '_0000.nii.gz'))
    shutil.copyfile(join(subfilepath, names[1]), join(save_imgpath, case + '_0003.nii.gz')) # t2
    shutil.copyfile(join(subfilepath, names[2]), join(save_imgpath, case + '_0002.nii.gz')) #t1ce
    shutil.copyfile(join(subfilepath, names[3]), join(save_imgpath, case + '_0001.nii.gz')) #t2

#%% make and save whole tumor label

save_wtpath = r'Brats\wtlabelsTr'
save_enpath = r'Brats\enlabelsTr'

names = os.listdir(save_gtpath)
names.sort()

total_num = 0
wt_num = 0
en_num = 0
tc_num = 0

for name in names:
    gt_nii = nb.load(join(save_gtpath, name))
    gt_data = gt_nii.get_data()
    wt = np.int16(gt_data>0)
    wt_num = wt_num+np.count_nonzero(wt)
    save_wtnii = nb.Nifti1Image(wt, gt_nii.affine, gt_nii.header)
    nb.save(save_wtnii, join(save_wtpath, name))
    
    
    tc = gt_data==1
    en = gt_data==4
    tc_num = tc_num + np.count_nonzero(tc)
    en_num = en_num + np.count_nonzero(en)
    total_num += wt.size
    
#%% conver brats seg label to 1-3

def convert_brats_seg(seg):
    new_seg = np.zeros_like(seg, seg.dtype)
    new_seg[seg == 1] = 3 #TC
    new_seg[seg == 2] = 1 # WT
    new_seg[seg == 4] = 2 # EN   
    return new_seg


def convert_seg2bratsgt(seg):
    new_seg = np.zeros_like(seg, seg.dtype)
    new_seg[seg == 1] = 2
    new_seg[seg == 2] = 4
    new_seg[seg == 3] = 1   
    return new_seg

def extract_EN_TC(seg):
    new_seg = np.zeros_like(seg, seg.dtype)
    new_seg[seg == 1] = 2 # TC
    new_seg[seg == 4] = 1 # EN  
    return new_seg

label_path = r'BraTS\ValEnsemble'
save_path = r'BraTS\Submit\ValEnsem1'

names = os.listdir(label_path)
names.sort()

for name in names:
    gt_nii = nb.load(join(label_path, name))
    gt_data = gt_nii.get_data()
    save_data = convert_seg2bratsgt(gt_data)
    save_nii = nb.Nifti1Image(save_data, gt_nii.affine, gt_nii.header)
    nb.save(save_nii, join(save_path, name))
    print(name, ' finished!')




#%% label analysis; count voxel number for each class
label_path = r''
names = os.listdir(label_path)
names.sort()
records = OrderedDict()
records['name'] = list()
records['wt_num'] = list()
records['en_num'] = list()
records['tc_num'] = list()

for name in names:
    gt_nii = nb.load(join(label_path, name))
    gt_data = gt_nii.get_data()
    wt = np.count_nonzero(gt_data>0)
    en = np.count_nonzero(gt_data>1)
    tc = np.count_nonzero(gt_data>2)
    records['name'].append(name)
    records['wt_num'].append(wt)
    records['en_num'].append(en)
    records['tc_num'].append(tc)

dataframe = pd.DataFrame(records)
dataframe.to_csv(r'\ValSeg1.csv', index=False)


#%% evaluate segmentation results
from medpy import metric

def brats_eval(seg, gt):
    wt_seg = seg>0; wt_gt = gt>0    
    et_seg = seg==2; et_gt = gt==2
    tc_seg = seg==3; tc_gt = gt==3
    wt_dice = round(metric.dc(wt_seg, wt_gt),4)
    if np.count_nonzero(et_gt) == 0:
        if np.count_nonzero(et_seg)>0:
            et_dice = 0
        else:
            et_dice = 1
    else:
        et_dice = round(metric.dc(et_seg, et_gt),4)
        
    if np.count_nonzero(tc_gt) == 0:
        if np.count_nonzero(tc_seg)>0:
            tc_dice = 0
        else:
            tc_dice = 1
    else:    
        tc_dice = round(metric.dc(tc_seg, tc_gt),4)
    
    return wt_dice, et_dice, tc_dice
    
seg_path = r''
label_path = r''
   
names = os.listdir(label_path)
names.sort()
records = OrderedDict()
records['name'] = list()
records['wt_num'] = list()
records['en_num'] = list()
records['tc_num'] = list()    
    
for name in names:
    seg_data = nb.load(join(seg_path, name)).get_data() 
    gt_data = nb.load(join(label_path, name)).get_data() 
    wt_dice, et_dice, tc_dice = brats_eval(seg_data, gt_data)
    records['name'].append(name)
    records['wt_num'].append(wt_dice)
    records['en_num'].append(et_dice)
    records['tc_num'].append(tc_dice)

dataframe = pd.DataFrame(records)
dataframe.to_csv(r'pathto\TrainSeg1.csv', index=False)
    
    

    
    


