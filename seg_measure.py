# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:43:25 2019
segmentation measure
@author: Jun Ma
"""

import numpy as np
import nibabel as nb
import os, re
from medpy import metric
from collections import OrderedDict
import pandas as pd
from skimage import measure

def getLargestCC(segmentation):
    labels = measure.label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC


#%% liver tumor in MR images

gt_path = r'labels'
seg_path = r'seg'

# liver tumor in CT images
if os.path.exists(os.path.join(seg_path, 'plans.pkl')):
    os.remove(os.path.join(seg_path, 'plans.pkl'))

print(seg_path)
seg_measures = OrderedDict()
seg_measures['image'] = list()    
seg_measures['Dice'] = list()
seg_measures['RAVD'] = list()
seg_measures['ASSD'] = list()
seg_measures['MSSD'] = list()


filenames = os.listdir(seg_path)
filenames.sort()

for name in filenames:
    name_sp = name.split(sep='_')
    name_num = re.findall('\d+', name_sp[1])[0]
#    name_num = name[3:5]
    gt_nii = nb.load(os.path.join(gt_path, name_sp[0] + '_' + name_num + '.nii.gz'))
    gt = gt_nii.get_data() # tumorgt
    vxlspacing = gt_nii.header.get_zooms()
    seg = nb.load(os.path.join(seg_path, name)).get_data()
    seg = np.squeeze(seg)
#    seg = np.uint8(seg>1)
    if np.max(seg)>0:
#        gt = gt>1
#        seg = getLargestCC(seg>0)
        seg_dice = round(metric.dc(seg>0, gt), 4)
        seg_RAVD = round(metric.ravd(seg, gt), 4)   
        seg_ASSD = round(metric.binary.assd(seg, gt, vxlspacing), 4)
        seg_MSSD = round(metric.hd(seg, gt, vxlspacing), 4)
    else:
        seg_dice = 0.0
        seg_RAVD = 0.0  
        seg_ASSD = 0.0
        seg_MSSD = 0.0
    
    
    seg_measures['image'].append(name)
    seg_measures['Dice'].append(seg_dice)
    seg_measures['RAVD'].append(seg_RAVD)
    seg_measures['ASSD'].append(seg_ASSD)
    seg_measures['MSSD'].append(seg_MSSD)
    print(name, seg_dice)
    
print('Average Dice: ', sum(seg_measures['Dice'])/len(filenames))
print('Average RAVD: ', sum(seg_measures['RAVD'])/len(filenames))
print('Average ASSD: ', sum(seg_measures['ASSD'])/len(filenames))
print('Average MSSD: ', sum(seg_measures['MSSD'])/len(filenames))

#%%
dataframe = pd.DataFrame(seg_measures)
dataframe.to_csv(r'results.csv', index=False)


