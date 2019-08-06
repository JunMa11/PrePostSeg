import nibabel as nb
import os
import re
from skimage import morphology, measure
import numpy as np
import scipy

# get largets region; Usually, it is used to extract the largest organ
def getLargestCC(segmentation):
    """segmentation: binary image"""
    labels = measure.label(segmentation)
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

# remove small holes or objects with in predefined size
liver_mask = morphology.remove_small_holes(liver_mask.astype(np.bool), 100000)  
liver_mask = morphology.remove_small_objects(liver_mask, min_size=64)






