# -*- coding: utf-8 -*-
               
import configparser
import argparse
import logging
import os
from nilearn import image
from matplotlib import pyplot as plt
import numpy as np
import time 
from utils import extract_timeseries, plot_meanTs
from refine import refine_roi
import sys

# PARSE ARGUMENTS AND CONFIGURATION FILE

parser = argparse.ArgumentParser(
    description='description of this parser object:'
    )
parser.add_argument(
    '-v','--verbose', help="be verbose (INFO level)",
    action="store_const", dest="loglevel", const=logging.INFO,
    )
parser.add_argument(
    '-d','--debug', help="print everything (DEBUG level)",
    action="store_const", dest="loglevel", const=logging.DEBUG,
    )
parser.add_argument(
    '-c', '--config', type=str, help="configuration file path (default value: 'config.INI')",
    default = 'config.INI'
    )
parser.add_argument(
    '-l', '--logfile', type=str, help="log file path (default value: None)",
    default = None
    )
args = parser.parse_args()


logging.basicConfig(filename=args.logfile, filemode='a', level=args.loglevel,
                    format = "%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger('matplotlib.font_manager').disabled = True

config = configparser.ConfigParser()
if os.path.isfile(args.config):
    try:
        config.read(args.config)
    except:
        raise ValueError(f"can't read configuration file {args.config}")
else:
    raise ValueError(f"can't find configuration file {args.config}")


# EXTRACT TIMESERIES FROM ALL THE VOXELS IN THE ROI

start_time = time.time()
print("---- rs-DATA PARCELLATION IMPROVEMENT algorithm (single ROI) ----")     

print("Loading timeseries from fData...")
logging.info(f"fData path: {config['REFINE_SINGLE_ROI.PY']['fData_path']}")
fData = image.get_data(config['REFINE_SINGLE_ROI.PY']['fData_path'])
ROImask = image.get_data(config['REFINE_SINGLE_ROI.PY']['ROImask_path'])

print("Extracting ROI timeseries...")
logging.info(f"ROI mask path: {config['REFINE_SINGLE_ROI.PY']['ROImask_path']}")
stSeries, ROImask, n = extract_timeseries(fData, ROImask,
                                          sigma = float(config['REFINE_SINGLE_ROI.PY']['sigma'])
                                          )
logging.info(f"% broken voxels: {n*100:.2f}")


# PLOT ORIGINAL MEAN TS AND COMPUTE SNR
fig_ts, ax_ts = plt.subplots(figsize=(6,2), tight_layout=True)
SNR = plot_meanTs(stSeries, ax=ax_ts, TR = 0.735,
                  shadeColor = 'grey', linewidth=1, color='black')


# REFINE ROI
print("Refining ROI...")
logging.info(f"Using quantile threshold = {config['REFINE_SINGLE_ROI.PY']['qTh']}")
ROImask_t, corrMap = refine_roi(stSeries,
                                ROImask,
                                onlyEdges = True,
                                quantileTh = float(config['REFINE_SINGLE_ROI.PY']['qTh']),
                                return_mode = 'over')
logging.info(f"ROI's volume (before/after): {np.sum(ROImask)}/{np.sum(ROImask_t)}")
logging.info(f"removed {100*(np.sum(ROImask)-np.sum(ROImask_t))/np.sum(ROImask):.3}%")


# PLOT NEW TIMESERIES
stSeries_new, _, _ = extract_timeseries(fData, ROImask_t,
                                        sigma = float(config['REFINE_SINGLE_ROI.PY']['sigma'])
                                        )
fig_ts_new, ax_ts_new = plt.subplots(figsize=(6,2), tight_layout=True)
SNR = plot_meanTs(stSeries_new, ax=ax_ts_new, TR = 0.735,
                  shadeColor = 'grey', linewidth=1, color='black')


# SAVE RESULTS

print("Saving results...")
logging.info(f"Results path: {config['REFINE_SINGLE_ROI.PY']['results_path']}")
img_newROI = image.new_img_like(image.load_img(config['REFINE_SINGLE_ROI.PY']['ROImask_path']),
                                ROImask_t.astype(int)) 
img_newROI.to_filename(f"{config['REFINE_SINGLE_ROI.PY']['results_path']}/newroi.nii.gz")
fig_ts.savefig(f"{config['REFINE_SINGLE_ROI.PY']['results_path']}/ts_OLD.jpg", format="jpg")
fig_ts_new.savefig(f"{config['REFINE_SINGLE_ROI.PY']['results_path']}/ts_NEW.jpg", format="jpg")


print(f"Finished!\nTotal elapsed time: {time.time()-start_time:.4}s ")

