# -*- coding: utf-8 -*-
"""
    usage:
    refine_single_ROI.py    -fData <path_to_fMRI_data >
                            (-ROImask <path_to_ROI_mask>)
    
"""



import argparse
import logging
import nilearn
from nilearn import image, plotting
from matplotlib import pyplot as plt
import scipy as sp
import numpy as np 
import time 
import os

parser = argparse.ArgumentParser(description='description of this parser object')
parser.add_argument(
    '-fData', type=str, help='path to functional data',
    default = 'F:/neurocovid/HC/18G00027/filtered_func_data_clean_toMNI.nii'
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


#%% EXTRACTING TIMESERIES FROM ALL THE VOXELS IN THE ROI

logging.debug(f"fData path: {args.fData})")
logging.debug(f"ROI mask path: {args.ROImask})")
logging.info("Extracting timeseries from fData...")


numpy_ROImask = image.get_data(args.ROImask)
numpy_fData = image.get_data(args.fData)
tSeries = numpy_fData[numpy_ROImask.astype(bool)]
for i in range(len(tSeries)):
    tSeries[i] = (tSeries[i]-np.mean(tSeries[i]))/np.std(tSeries[i])
tSeries = tSeries.T


# broken voxels have timeseries == 0 for the whole rs length
broken_voxels = np.all(tSeries<1e-6, axis=0).astype(int)

logging.debug(f"N. broken voxels: {len(np.where(broken_voxels)[0])}")
logging.debug(f"N. voxels in the ROI: {len(tSeries.T)}")

#%% GAUSSIAN FILTERING OF THE TIMESERIES

logging.info("Gaussian smoothing of timeseries...")
stSeries = [ sp.ndimage.gaussian_filter1d(tSeries[:,j], sigma=args.sigma)
                for j in range(np.shape(tSeries)[1])]
stSeries = np.asarray(stSeries).transpose()

#%% PLOT ORIGINAL MEAN TS AND COMPUTE SNR

ts_m = np.average(stSeries.T, axis=0)
ts_s = np.std(stSeries.T, axis=0)
SNR = np.mean(np.abs(ts_m/ts_s))
            
fig_ts, ax_ts = plt.subplots(figsize=(7.5,2), dpi=300, tight_layout=True)
ax_ts.plot(np.arange(len(ts_m))*0.735, ts_m, '-', linewidth=1, color="black")
ax_ts.fill_between(np.arange(len(ts_m))*0.735, (ts_m-ts_s), (ts_m+ts_s), color='grey', alpha=.4)
ax_ts.set_xlim([0,600])
ax_ts.set_ylim([-2.3,2.3])
ax_ts.set_xlabel("Time (s)")
ax_ts.set_title(f"avg SNR: {np.mean(np.abs(np.asarray(ts_m)/np.asarray(ts_s))):.5f}")
    
#%% REFINING ROI

# internal correlation matrix 
corrMat = np.corrcoef(stSeries, rowvar=False)
np.fill_diagonal(corrMat,0)

# avg corrcoef values
avg_corrMat = np.mean(corrMat, axis=0)
avg_corrMat[avg_corrMat<0] = 0

# back-project to 3D corr map
numpy_corrMap = np.zeros_like(numpy_ROImask)
n=0
for x in range(np.shape(numpy_ROImask)[0]):
    for y in range(np.shape(numpy_ROImask)[1]):
        for z in range(np.shape(numpy_ROImask)[2]):
            if numpy_ROImask[x,y,z]:
                numpy_corrMap[x,y,z] = avg_corrMat[n]
                n+=1

numpy_edge = np.logical_xor(numpy_ROImask,
                           sp.ndimage.binary_erosion(numpy_ROImask))

for x in range(np.shape(numpy_ROImask)[0]):
    if np.any(numpy_ROImask[x,:,:]):
        fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(10,3), dpi=300, tight_layout=True)
        ax[0].imshow(numpy_ROImask[x,:,:], cmap = 'Greys')
        ax[0].set_title("Original ROI mask")
        ax[1].imshow(numpy_edge[x,:,:], cmap = 'Greys')
        ax[1].set_title("Edges")
        im = ax[2].imshow(numpy_corrMap[x,:,:], cmap = 'YlOrRd')
        ax[2].set_title("Correlation map")
        fig.colorbar(im)
        
        