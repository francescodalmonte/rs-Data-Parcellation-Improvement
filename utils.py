# -*- coding: utf-8 -*-

import os
import logging

import numpy as np
import scipy as sp

from nilearn import image


def load_timeseries(fData, ROImask, sigma=0):
    """
        Load timeseries from EPI
        
        ---- Parameters ----
        fData           :   path to EPI image
        ROImask         :   path to ROI mask or path to ROIs masks directory
        sigma           :   sigma for ts Gaussian smoothing 
                            (default=0, i.e. no smoothing)
        
        ---- Returns ----
        tSeries         :   timeseries (numpy array)
        numpy_fData     :   original EPI data (numpy array)
        numpy_ROImask   :   mask (numpy array)
        broken_voxels   :   broken voxels
        
        
    """
       
    logging.debug(f"fData path: {fData}")     
    logging.info("Extracting timeseries from fData...")

    numpy_fData = image.get_data(fData)

    if os.path.isfile(ROImask):  
        logging.debug(f"ROI mask path: {ROImask}")         
        numpy_ROImask = image.get_data(ROImask)
            
        tSeries = numpy_fData[numpy_ROImask.astype(bool)]
        
        # broken voxels have timeseries == 0 for the whole rs length
        broken_voxels = np.all(tSeries<1e-6, axis=1)
        tSeries = tSeries[np.logical_not(broken_voxels)]
        logging.debug(f"N. broken voxels: {len(np.where(broken_voxels)[0])}")
        logging.debug(f"N. voxels in the ROI: {len(tSeries)}")
        
        for i in range(len(tSeries)):
            tSeries[i] = (tSeries[i]-np.mean(tSeries[i]))/np.std(tSeries[i])
        tSeries = tSeries.T

        if sigma>1e-6:
            logging.info("Gaussian smoothing of timeseries...")
            logging.debug(f"Using sigma = {sigma}")
            tSeries = [ sp.ndimage.gaussian_filter1d(tSeries[:,j], sigma=sigma)
                        for j in range(np.shape(tSeries)[1])]
            tSeries = np.asarray(tSeries).transpose()
        
        return tSeries, numpy_fData, numpy_ROImask, broken_voxels
    
    elif os.path.isdir(ROImask):  
        logging.debug(f"ROIs dir path: {ROImask}") 



    




