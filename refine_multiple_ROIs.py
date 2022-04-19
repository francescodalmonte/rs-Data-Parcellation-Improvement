# -*- coding: utf-8 -*-
"""
    usage:
    refine_single_ROI.py    -fData <path_to_fMRI_data >
                            (-MNImask <path_to_MNI_brain_mask>)
    
"""



import argparse
import logging
import nilearn
from nilearn import maskers
import scipy as sp
import numpy as np 
import time 
import os

parser = argparse.ArgumentParser(description='description of this parser object')
parser.add_argument(
    '-fData', type=str, help='path to functional data',
    default = 'F:/neurocovid/HC/18G00027/filtered_func_data_clean_toMNI.nii.gz'
    )
parser.add_argument(
    '-MNImask', type=str, help='path to MNI brain mask',
    default = 'std/MNI152_T1_2mm_brain_mask.nii.gz'
    )
parser.add_argument(
    '-ROImask', type=str, help='path to ROI mask',
    default = 'rois/roi_00.nii.gz'
    )
parser.add_argument(
    '-sigma', type=float, help='timeseries Gaussian smoothing sigma value',
    default = 1
    )
parser.add_argument(
    '-verbose', help="be verbose",
    action="store_const", dest="loglevel", const=logging.INFO,
    )
parser.add_argument(
    '-debug', help="print all debugging statements",
    action="store_const", dest="loglevel", const=logging.DEBUG,
    default=logging.WARNING,
    )

args = parser.parse_args()    
logging.basicConfig(level=args.loglevel,
                    format = "%(message)s")


#%% EXTRACTING TIMESERIES FROM ALL THE VOXELS IN GRAY MATTER

logging.debug(f"fData path: {args.fData})")
logging.info("Extracting timeseries from fData...")

brain_masker = maskers.NiftiMasker(mask_img = args.MNImask, t_r=0.735)
time_series = brain_masker.fit_transform(args.fData)

# broken voxels have timeseries == 0 for the whole rs length
broken_voxels = np.all(time_series<1e-6, axis=0).astype(int)

logging.debug(f"N. broken voxels: {len(np.where(broken_voxels)[0])}")

#%% GAUSSIAN FILTERING OF THE TIMESERIES

logging.info("Gaussian smoothing of timeseries...")
stime_series = [ sp.ndimage.gaussian_filter1d(time_series[:,j], sigma=args.sigma)
                for j in range(np.shape(time_series)[1])]
stime_series = np.asarray(stime_series).transpose()

#%% EXTRACTING ROIs MASKS

logging.debug(f"ROI msk path: {args.ROImask})")
logging.info("Extracting ROI mask...")

roi_mask = brain_masker.transform(args.ROImask)[0]

# removing broken voxels
roi_mask = roi_mask*np.logical_not(broken_voxels).astype(int)
logging.debug(f"N. voxels in the ROI: {(len(broken_voxels)-sum(roi_mask==0)).astype(int)}")

