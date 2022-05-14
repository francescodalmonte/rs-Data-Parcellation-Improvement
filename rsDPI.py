# -*- coding: utf-8 -*-

import os
import time 
import sys    
import configparser
import argparse
import logging

from nilearn import image
from matplotlib import pyplot as plt
import numpy as np

from utils import extract_timeseries, plot_meanTs, ts_stats
from refine import refine_roi



def setup_args():
    parser = argparse.ArgumentParser(
        description='Refine an existing segmentation of functionally connected brain regions'
        )
    parser.add_argument(
        '-m','--multiROI', help="run in multiROI mode",
        action="store_true"
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
        '-p', '--savePlots', help="save .jpeg plots to file",
        action = 'store_true'
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
    config = configparser.ConfigParser()
    if os.path.isfile(args.config):
        try:
            config.read(args.config)
        except:
            raise ValueError(f"can't read configuration file {args.config}")
    else:
        raise ValueError(f"can't find configuration file {args.config}")
    
    return args, config



def setup_logging(args):
    """
        Setup logging behaviour
    """
    logging.basicConfig(filename=args.logfile, filemode='a', level=args.loglevel,
                        format = "%(asctime)s - %(levelname)s - %(message)s")
    logging.getLogger('matplotlib.font_manager').disabled = True



def main_singleMode(args, config):
    """
        Run algorithm for a single-ROI 
        
    """
    start_time = time.time()
    logging.info("rsDATA PARCELLATION IMPROVEMENT algorithm - (single ROI)")    

    # EXTRACT TIMESERIES
    logging.info("Loading data from EPI...")
    
    logging.debug(f"fData path: {config['fData_path']}")
    logging.debug(f"ROI mask path: {config['ROI_path']}")
    
    fData = image.get_data(config['fData_path'])
    ROImask = image.get_data(config['ROI_path'])

    logging.info("Extracting ROI timeseries and plotting...")
    stSeries, ROImask, n = extract_timeseries(fData, ROImask,
                                              sigma = float(config['sigma'])
                                              )
    
    if args.savePlots:
        # PLOT ORIGINAL MEAN TS AND COMPUTE SNR
        fig_ts, ax_ts = plt.subplots(figsize=(6,2), tight_layout=True)
        _ = plot_meanTs(stSeries, ax=ax_ts, TR = 0.735,
                        shadeColor = 'grey', linewidth=1, color='black')


    # REFINE ROI
    logging.info("Refining ROI...")
    ROImask_t, corrMap = refine_roi(stSeries,
                                    ROImask,
                                    onlyEdges = True,
                                    quantileTh = float(config['qTh']),
                                    )
    
    if args.savePlots:
        # PLOT NEW TIMESERIES
        logging.info("Extracting new ROI timeseries and plotting...")
        stSeries_new, _, _ = extract_timeseries(fData, ROImask_t,
                                            sigma = float(config['sigma'])
                                            )
        fig_ts_new, ax_ts_new = plt.subplots(figsize=(6,2), tight_layout=True)
        _ = plot_meanTs(stSeries_new, ax=ax_ts_new, TR = 0.735,
                    shadeColor = 'grey', linewidth=1, color='black')
    

    # SAVE RESULTS

    logging.info("Saving results...")
    logging.debug(f"Results path: {config['results_dirpath']}")
    img_newROI = image.new_img_like(image.load_img(config['ROI_path']),
                                    ROImask_t.astype(int)) 
    img_newROI.to_filename(f"{config['newROI_dirpath']}/newroi.nii.gz")
    if args.savePlots:
        fig_ts.savefig(f"{config['results_dirpath']}/ts_OLD.jpg", format="jpg")
        fig_ts_new.savefig(f"{config['results_dirpath']}/ts_NEW.jpg", format="jpg")
    
    logging.info(f"Finished!\nTotal elapsed time: {time.time()-start_time:.4}s ")




def main_multiMode(args, config):
    """
        Run algorithm for a complete brain parcellation (set of ROIs) 
        
    """
    start_time = time.time()
    logging.info("rsDATA PARCELLATION IMPROVEMENT algorithm - (multi ROI)")    

    if not os.path.isdir(config['newROI_dirpath']):
        os.mkdir(config['newROI_dirpath'])

    # EXTRACT TIMESERIES
    logging.info("Loading data from EPI...")
    fData = image.get_data(config['fData_path'])


    files = os.listdir(config['ROI_dirpath'])
    for filename in files:
        if filename[-7:] == '.nii.gz':
            singleStart = time.time()

            ROImask = image.get_data(f"{config['ROI_dirpath']}/{filename}")
            
            # EXTRACT TIMESERIES
            logging.info(f"{filename} : Extracting timeseries...")
            stSeries, ROImask, _ = extract_timeseries(fData, ROImask,
                                                      sigma = float(config['sigma'])
                                                      )
                                                      
            # REFINE ROI
            logging.info(f"{filename} : ROI refining...")
            ROImask_t, corrMap = refine_roi(stSeries,
                                            ROImask,
                                            onlyEdges = True,
                                            quantileTh = float(config['qTh'])
                                            )
            
            if args.savePlots:
                # EXTRACT NEW TIMESERIES
                logging.info(f"{filename} : Extracting new timeseries...")
                stSeries_t, ROImask_t, _ = extract_timeseries(fData, ROImask_t,
                                                          sigma = float(config['sigma'])
                                                          )

            # SAVE RESULTS
            img_newROI = image.new_img_like(image.load_img(f"{config['ROI_dirpath']}/{filename}"),
                                    ROImask_t.astype(int)) 
            img_newROI.to_filename(f"{config['newROI_dirpath']}/new_{filename}")

            if args.savePlots:
                fig, ax = plt.subplots(figsize=(6,2), dpi=300, tight_layout=True)
                plot_meanTs(stSeries, ax=ax, TR = 0.735, shadeColor = 'grey',
                        linewidth=1, color='black')
                ax.set_ylim([-2.5,2.5])
                fig.savefig(f"{config['results_dirpath']}/{filename[:-7]}_ts.jpg", format="jpg")

                fig, ax = plt.subplots(figsize=(6,2), dpi=300, tight_layout=True)
                plot_meanTs(stSeries_t, ax=ax, TR = 0.735, shadeColor = 'grey',
                        linewidth=1, color='black')
                ax.set_ylim([-2.5,2.5])
                fig.savefig(f"{config['results_dirpath']}/{filename[:-7]}_newts.jpg", format="jpg")
            
            logging.info(f"{filename} : Elapsed {(time.time()-singleStart):.4}s")
            
    logging.info(f"Finished!\nTotal elapsed time {time.time()-start_time:.4}s ")

          


          
def main():
    # PARSE ARGUMENTS AND CONFIGURATION FILE
    args, config = setup_args()
    
    # SETUP LOGGING 
    setup_logging(args)
    
    if args.multiROI:
        main_multiMode(args, config['multiMode'])
    else: 
        main_singleMode(args, config['singleMode'])

if __name__=='__main__':
    main()
