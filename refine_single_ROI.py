# -*- coding: utf-8 -*-
"""
    usage:
    refine_single_ROI.py    -fData <path_to_fMRI_data >
                            (-ROImask <path_to_ROI_mask>)
                            
"""

import argparse
import logging
from nilearn import image
from matplotlib import pyplot as plt
import scipy as sp
import numpy as np 
import time 

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
    '-saveTo', type=str, help='path to results directory',
    default = 'results'
    )
parser.add_argument(
    '-sigma', type=float, help='timeseries Gaussian smoothing sigma value',
    default = 1
    )
parser.add_argument(
    '-qTh', type=float, help='quantile threshold',
    default = 0.25
    )
parser.add_argument(
    '-v','-verbose', help="be verbose",
    action="store_const", dest="loglevel", const=logging.INFO,
    )
parser.add_argument(
    '-d','-debug', help="print all debugging statements",
    action="store_const", dest="loglevel", const=logging.DEBUG,
    default=logging.WARNING,
    )

args = parser.parse_args()    
logging.basicConfig(level=args.loglevel,
                    format = "%(message)s")
logging.getLogger('matplotlib.font_manager').disabled = True


#%% EXTRACTING TIMESERIES FROM ALL THE VOXELS IN THE ROI

start_time = time.time()
logging.info("---------- rs-DATA PARCELLATION IMPROVEMENT algorithm (single ROI)")

logging.debug(f"fData path: {args.fData}")
logging.debug(f"ROI mask path: {args.ROImask}")
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
logging.debug(f"Using sigma = {args.sigma}")

stSeries = [ sp.ndimage.gaussian_filter1d(tSeries[:,j], sigma=args.sigma)
                for j in range(np.shape(tSeries)[1])]
stSeries = np.asarray(stSeries).transpose()

#%% PLOT ORIGINAL MEAN TS AND COMPUTE SNR

ts_m = np.average(stSeries.T, axis=0)
ts_s = np.std(stSeries.T, axis=0)
SNR = np.mean(np.abs(ts_m/ts_s))
            
fig_ts, ax_ts = plt.subplots(figsize=(6,2), tight_layout=True)
ax_ts.plot(np.arange(len(ts_m))*0.735, ts_m, '-', linewidth=1, color="black")
ax_ts.fill_between(np.arange(len(ts_m))*0.735, (ts_m-ts_s), (ts_m+ts_s), color='grey', alpha=.4)
ax_ts.set_xlim([0,600])
ax_ts.set_ylim([-2.3,2.3])
ax_ts.set_xlabel("Time (s)")
ax_ts.set_title(f"avg SNR: {np.mean(np.abs(np.asarray(ts_m)/np.asarray(ts_s))):.5f}")
    
#%% REFINING ROI

logging.info("Refining ROI...")
logging.debug(f"Using quantile threshold = {args.qTh}")

# internal correlation matrix 
corrMat = np.corrcoef(stSeries, rowvar=False)
np.fill_diagonal(corrMat,0)

# avg corrcoef values
avg_corrMat = np.mean(corrMat, axis=0)
avg_corrMat[avg_corrMat<0] = 0
logging.debug(f"min/max average corr. values: {np.min(avg_corrMat):.4}/{np.max(avg_corrMat):.4}")

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
threshold_value = np.quantile(avg_corrMat, args.qTh)
numpy_threshold = np.ones_like(numpy_corrMap)*threshold_value
numpy_cut = np.logical_and(numpy_corrMap<numpy_threshold,
                           numpy_edge)
numpy_ROImask_th = numpy_ROImask*np.logical_not(numpy_cut)

logging.debug(f"ROI's volume (before/after): {np.sum(numpy_ROImask)}/{np.sum(numpy_ROImask_th)}")
logging.debug(f"removed {100*(np.sum(numpy_ROImask)-np.sum(numpy_ROImask_th))/np.sum(numpy_ROImask):.3}%")

#%% PLOT NEW TIMESERIES, PLOT RESULTS
logging.info("New timeseries extraction...")

# timeseries
tSeries_new = numpy_fData[numpy_ROImask_th.astype(bool)]
for i in range(len(tSeries_new)):
    tSeries_new[i] = (tSeries_new[i]-np.mean(tSeries_new[i]))/np.std(tSeries_new[i])
tSeries_new = tSeries_new.T

stSeries_new = [ sp.ndimage.gaussian_filter1d(tSeries_new[:,j], sigma=args.sigma)
                for j in range(np.shape(tSeries_new)[1])]
stSeries_new = np.asarray(stSeries_new).transpose()

ts_m_new = np.average(stSeries_new.T, axis=0)
ts_s_new = np.std(stSeries_new.T, axis=0)
SNR_new = np.mean(np.abs(ts_m_new/ts_s_new))
            
fig_ts_new, ax_ts_new = plt.subplots(figsize=(6,2), tight_layout=True)
ax_ts_new.plot(np.arange(len(ts_m_new))*0.735, ts_m_new, '-',
           linewidth=1, color="black")
ax_ts_new.fill_between(np.arange(len(ts_m_new))*0.735,
                   (ts_m_new-ts_s_new), (ts_m_new+ts_s_new), 
                   color='grey', alpha=.4)
ax_ts_new.set_xlim([0,600])
ax_ts_new.set_ylim([-2.3,2.3])
ax_ts_new.set_xlabel("Time (s)")
ax_ts_new.set_title(f"avg SNR (NEW): {np.mean(np.abs(np.asarray(ts_m_new)/np.asarray(ts_s_new))):.5f}")

# new roi
levels = []
for x in range(np.shape(numpy_ROImask)[0]):
    if np.any(numpy_ROImask[x,:,:]):
        levels.append(x)
        
fig_ROI, ax_ROI = plt.subplots(nrows=len(levels), ncols=3, figsize=(7,2*len(levels)), tight_layout=True)
for l in range(len(levels)):
    ax_ROI[l,0].imshow(numpy_ROImask[levels[l],:,:], cmap = 'Blues')
    ax_ROI[l,0].set_title(f"Original ROI (lev.{levels[l]})")
    ax_ROI[l,0].set_xticklabels([]); ax_ROI[l,0].set_yticklabels([])
    ax_ROI[l,1].imshow(numpy_cut[levels[l],:,:], cmap = 'Blues')
    ax_ROI[l,1].set_title("Removed")
    ax_ROI[l,1].set_xticklabels([]); ax_ROI[l,1].set_yticklabels([])
    ax_ROI[l,2].imshow(numpy_ROImask_th[levels[l],:,:], cmap = 'Blues')
    ax_ROI[l,2].set_title("Final ROI")    
    ax_ROI[l,2].set_xticklabels([]); ax_ROI[l,2].set_yticklabels([])


#%% SAVE RESULTS
logging.info("Saving results...")

img_newROI = image.new_img_like(image.load_img(args.ROImask), numpy_ROImask_th.astype(int)) 
img_newROI.to_filename(f"{args.saveTo}/newroi.nii.gz")

fig_ts.savefig(f"{args.saveTo}/ts_OLD.jpg", format="jpg")
fig_ts_new.savefig(f"{args.saveTo}/ts_NEW.jpg", format="jpg")

fig_ROI.savefig(f"{args.saveTo}/roi.jpg", format="jpg")

logging.info(f"---------- Total elapsed time {time.time()-start_time:.4}s ")

