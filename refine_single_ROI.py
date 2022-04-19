# -*- coding: utf-8 -*-

import nilearn
from nilearn import input_data
import scipy as sp
import numpy as np 
import time 
import os


#%% EXTRACTING TIMESERIES FROM ALL THE VOXELS IN GRAY MATTER

func_data_path = "F:/neurocovid/HC/21G29681/filtered_func_data_clean_toMNI.nii.gz"
MNI_path = "C:/Users/Francesco/thesis_neurocovid/Data/MNI152_T1_2mm.nii.gz"
MNI_mask_path = "C:/Users/Francesco/thesis_neurocovid/Data/MNI152_T1_2mm_brain_mask.nii.gz"

print("Extracting timeseries...")
brain_masker = input_data.NiftiMasker(mask_img = MNI_mask_path, 
                            standardize=True, t_r=0.735)
time_series = brain_masker.fit_transform(func_data_path)

# broken voxels have timeseries == 0 for the whole rs length
broken_voxels = np.all(time_series == 0, axis=0).astype(int)
print(f"N. broken voxels: {len(np.where(broken_voxels)[0])}")
