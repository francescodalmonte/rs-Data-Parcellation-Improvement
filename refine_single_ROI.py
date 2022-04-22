# -*- coding: utf-8 -*-
               
import argparse
import logging
from nilearn import image
from matplotlib import pyplot as plt
import numpy as np
import time 
from utils import extract_timeseries, plot_meanTs
from refine import refine_roi


# ssostituire con configuration file
parser = argparse.ArgumentParser(
    description='description of this parser object'
    )
parser.add_argument(
    '-fData', type=str, help='path to functional data',
    default = 'F:/neurocovid/HC/18G00027/filtered_func_data_clean_toMNI.nii.gz'
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
    default = None
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
logging.info("Loading timeseries from fData...")

fData = image.get_data(args.fData)
ROImask = image.get_data(args.ROImask)

logging.info("Extracting ROI timeseries...")
stSeries, ROImask, n = extract_timeseries(fData, ROImask, sigma = args.sigma)
logging.debug(f"% broken voxels: {n*100:.2f}")

#%% PLOT ORIGINAL MEAN TS AND COMPUTE SNR

logging.info("Plotting timeseries and computing average SNR...")

fig_ts, ax_ts = plt.subplots(figsize=(6,2), tight_layout=True)
SNR = plot_meanTs(stSeries, ax=ax_ts, TR = 0.735,
                  shadeColor = 'grey', linewidth=1, color='black')


#%% REFINING ROI
logging.info("Refining ROI...")
logging.debug(f"Using quantile threshold = {args.qTh}")

ROImask_t, corrMap = refine_roi(stSeries,
                                ROImask,
                                onlyEdges = True,
                                quantileTh = args.qTh)


logging.debug(f"ROI's volume (before/after): {np.sum(ROImask)}/{np.sum(ROImask_t)}")
logging.debug(f"removed {100*(np.sum(ROImask)-np.sum(ROImask_t))/np.sum(ROImask):.3}%")

#%% PLOT NEW TIMESERIES, PLOT RESULTS
logging.info("New timeseries extraction and plotting...")

# new timeseries
stSeries_new, _, _ = extract_timeseries(fData, ROImask_t, sigma = args.sigma)

# PLOT NEW MEAN TS
fig_ts_new, ax_ts_new = plt.subplots(figsize=(6,2), tight_layout=True)
SNR = plot_meanTs(stSeries_new, ax=ax_ts_new, TR = 0.735,
                  shadeColor = 'grey', linewidth=1, color='black')

#%% SAVE RESULTS
logging.info("Saving results...")

img_newROI = image.new_img_like(image.load_img(args.ROImask),
                                ROImask_t.astype(int)) 
img_newROI.to_filename(f"{args.saveTo}/newroi.nii.gz")

fig_ts.savefig(f"{args.saveTo}/ts_OLD.jpg", format="jpg")
fig_ts_new.savefig(f"{args.saveTo}/ts_NEW.jpg", format="jpg")


logging.info(f"---------- Total elapsed time {time.time()-start_time:.4}s ")

