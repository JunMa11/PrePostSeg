# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:03:22 2019

@author: Jun
"""

from collections import OrderedDict
#import numpy as np
#from scipy.ndimage import label
import os
import json

join = os.path.join


#test_dir = r'\test'
label_dir = r'\nnUNet_raw_splitted\Task00_RoiMRTS\labelsTr'
output_folder = r'\data\nnUNet_raw_splitted\Task00_RoiMRTS'

train_ids=[]
test_ids = []
filenames = os.listdir(label_dir)
filenames.sort()
for name in filenames:
    train_ids.append(name.split('.nii.gz')[0])

add_test_id = True
if add_test_id:
    testnames = os.listdir(os.path.join(output_folder,'imagesTs'))
    testnames.sort()
    for test_name in testnames:
        test_ids.append(test_name.split('_0000.nii.gz')[0])

# manually set
json_dict = OrderedDict()
json_dict['name'] = "Task00_RoiMRTS"
json_dict['description'] = "Pancreas segmentation"
json_dict['tensorImageSize'] = "4D"
json_dict['reference'] = "f off, this is private"
json_dict['licence'] = "touch it and you die"
json_dict['release'] = "0.0"
json_dict['modality'] = {
    "0": "MR"
#    "1": "T2",
#    "2": "T1ce",
#    "3": "T1"
}
# manually set
json_dict['labels'] = {
    "0": "background",
    "1": "liver",
    "2": "tumor"
}

json_dict['numTraining'] = len(train_ids)
json_dict['numTest'] = len(test_ids)
json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in train_ids]
json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_ids]

with open(os.path.join(output_folder, "dataset.json"), 'w') as f:
    json.dump(json_dict, f, indent=4, sort_keys=True)

#%% show json information
file = r'pathto\splits_final.pkl'
import pickle, os
with open(file, 'rb') as f:
    data=pickle.load(f) 

for i in range(5):
    for j in range(60):
        data[i]['train'][j] = 'img_' + data[i]['train'][j][-3:]
    for k in range(15):
        data[i]['val'][k] = 'img_' + data[i]['val'][k][-3:]
        

output_folder = r'MultiSeries'
with open(os.path.join(output_folder, "splits_final.pkl"), 'wb') as f:
    pickle.dump(data, f)

#%% modify CNN parameters
file = r'KITS\nnUNetPlans_plans_3D.pkl'
import pickle, os
with open(file, 'rb') as f:
    data = pickle.load(f)

#%%
#data['base_num_features'] = 32
output_folder = r'\pre_data'
with open(os.path.join(output_folder, "nnUNetPlans_plans_2D.pkl"), 'wb') as f:
    pickle.dump(data2, f)


#%%
T1CNN_para = data['plans_per_stage'][0]
NewCNN_para = data2['plans_per_stage'][0]

NewCNN_para['conv_kernel_sizes'] = T1CNN_para['conv_kernel_sizes']
NewCNN_para['num_pool_per_axis'] = T1CNN_para['num_pool_per_axis']
NewCNN_para['pool_op_kernel_sizes'] = T1CNN_para['pool_op_kernel_sizes']










