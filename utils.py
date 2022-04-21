# -*- coding: utf-8 -*-

import os
import logging

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

from nilearn import image


def extract_timeseries(fData, ROImask, sigma = None):
    """
        Load timeseries from EPI
        
        ---- Parameters ----
        fData           :   path to EPI data or timeseries set
        ROImask         :   path to ROI mask file or ROI mask numpy array
        sigma           :   sigma for ts Gaussian smoothing 
                            (default=None, i.e. no smoothing)
        
        ---- Returns ----
        tSeries         :   ROI timeseries set (numpy array)
        broken_voxels   :   broken voxels
        
    """
    if not isinstance(fData, np.ndarray):
        if os.path.isfile(fData):
            fData = image.get_data(fData)
    if not isinstance(ROImask, np.ndarray):   
        if os.path.isfile(ROImask):  
            ROImask = image.get_data(ROImask)
            
    tSeries = fData[ROImask.astype(bool)]
        
    # broken voxels have timeseries == 0 for the whole rs length
    broken_voxels = np.all(tSeries<1e-6, axis=1)
    tSeries = tSeries[np.logical_not(broken_voxels)]
        
    for i in range(len(tSeries)):
        tSeries[i] = (tSeries[i]-np.mean(tSeries[i]))/np.std(tSeries[i])
    tSeries = tSeries.T

    if sigma is not None:
        tSeries = [ sp.ndimage.gaussian_filter1d(tSeries[:,j], sigma=sigma)
                   for j in range(np.shape(tSeries)[1])]
        tSeries = np.asarray(tSeries).transpose()
        
    return tSeries, broken_voxels
    



def plot_meanTs(tSeries, ax=None, TR = 1, shadeColor = 'white', **plt_kwargs):
    """
        Plot average timeseries of a signals set
        
        ---- Parameters ----
        tSeries         :   timeseries set
        ax              :   matplotlib axes to use (defult = None)
        TR              :   sampling period (to scale x-axis, default = 1) 
        shadeColor      :   1-std deviation interval color

        
        ---- Returns ----
        SNR             :   average signal-noise ratio
        
    """
    
    if ax is None:
        ax = plt.gca()
        
    ts_m = np.average(tSeries, axis=1)
    ts_s = np.std(tSeries, axis=1)
    SNR = np.mean(np.abs(ts_m/ts_s))
    
    ax.plot(np.arange(len(ts_m))*TR, ts_m, '-', **plt_kwargs)
    ax.fill_between(np.arange(len(ts_m))*TR,
                   (ts_m-ts_s), (ts_m+ts_s), color=shadeColor, alpha=.4)
    ax.set_ylim([-2.5,2.5])
    ax.set_xlabel("Time (s)")
    ax.set_title(f"avg SNR: {SNR:.5f}")
    return SNR




def tsSNR(tSeries):
    """
        Estimate signal-noise ratio of the average of a set of signals
        
        ---- Parameters ----
        tSeries         :   timeseries set
        
        ---- Returns ----
        SNR             :   average signal-noise ratio
        
    """

    ts_m = np.average(tSeries, axis=1)
    ts_s = np.std(tSeries, axis=1)
    SNR = np.mean(np.abs(ts_m/ts_s))
    
    return SNR





