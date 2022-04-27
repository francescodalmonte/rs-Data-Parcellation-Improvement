# -*- coding: utf-8 -*-

import os
import logging

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from nilearn import image



def back_project(array1D, ROImask):
    """
        Back-projects a 1D array onto a 3D array using a mask.
                
        Parameters
        ----------
        array1D     : 1D array
        ROImask     : binary ROI mask 
        
        Returns
        ----------
        map3D       : 3D re-mapping of x
        
        Notes:
        This function is used in refine_roi algorithm to re-map 
        a 1D array of average correlation values of the voxels
        of a ROI onto the 3D original space.
    """
    array1D = np.asarray(array1D)
    ROImask = np.asarray(ROImask)
    
    if np.sum(ROImask)!=len(array1D):
        raise ValueError("x length must match ROI size")
    map3D = np.zeros_like(ROImask)
    n=0
    for x in range(np.shape(ROImask)[0]):
        for y in range(np.shape(ROImask)[1]):
            for z in range(np.shape(ROImask)[2]):
                if ROImask[x,y,z]:
                    map3D[x,y,z] = array1D[n]
                    n+=1
    return map3D


def remove_broken_voxels(tSeries, ROImask=None, threshold = 1e-6):
    """
        Removes timeseries with no signal from a timeseries set,
        and removes corresponding voxels from ROImask (if provided)
        
        Parameters
        ----------
        tSeries     :   TxN timeseries set (T = n.timepoints,
                                            N = n.samples)
        ROImask     :   corresponding binary mask (optional)
        threshold   :   zero-signal threshold value
        
        Returns
        ----------
        tSeries_clean   :   cleaned timeseries set
        ROImask_clean   :   cleaned mask (optional)
        n               :   fraction of removed samples
    """
    tSeries = np.asarray(tSeries)
    if np.size(tSeries.T)==0: 
        raise ValueError("input tSeries is empty!")
        
    try: 
        np.shape(tSeries)[1]
    except: 
        raise ValueError("invalid tSeries argument ")
        
    if np.shape(tSeries)[1]<=1:
        raise ValueError("input tSeries has <=1 timepoints!")        
    
    
    # voxels with timeseries == 0 for the whole rs length
    broken_1 = np.all(np.abs(tSeries)<threshold, axis=0).astype(int)
    # voxels with NaN values
    broken_2 = np.any(tSeries!=tSeries, axis=0).astype(int)
    # broken voxels
    broken_voxels = np.logical_or(broken_1, broken_2)
    

    if not np.any(broken_voxels):
        logging.debug(f"no broken voxels")
        if ROImask is None:
            return tSeries.copy(), 0
        else:
            return tSeries.copy(), ROImask, 0
    else:
        # remove corresponding ts
        tSeries_clean = tSeries[:,np.logical_not(broken_voxels)].copy()
        # fraction of broken voxels
        n = (len(tSeries.T)-len(tSeries_clean.T))/len(tSeries.T)
        logging.debug(f"% broken voxels: {n*100:.2f}")
        if ROImask is None:
            return tSeries_clean, n            
        else:
            # remove voxels from ROImask
            ROImask_clean = ROImask - back_project(broken_voxels, ROImask)
            return tSeries_clean, ROImask_clean, n



def extract_timeseries(fData, ROImask, sigma = None, standardize = True):
    """
        Load timeseries from EPI
        
        Parameters
        ----------
        fData       :   path to EPI data or timeseries set
        ROImask     :   path to ROI mask file or ROI mask numpy array
        sigma       :   sigma for ts Gaussian smoothing 
                            (default=None, i.e. no smoothing)
        
        Returns
        ---------
        stSeries_c      :   TxN cleaned/smoothed ROI's timeseries set
                            (i.e. without broken voxels)
                            (T = n.timepoints, N = n.samples)                            
        ROImask_c       :   cleaned ROImask 
        n               :   fraction of broken voxels
        
    """
    
    if not isinstance(fData, np.ndarray):
        if os.path.isfile(fData):
            logging.debug(f"extracting_timeseries: argument fData is path: getting data from {fData}")
            fData = image.get_data(fData)
    if not isinstance(ROImask, np.ndarray):   
        if os.path.isfile(ROImask):  
            logging.debug(f"extracting_timeseries: argument ROImask is path: getting data from {ROImask}")
            ROImask = image.get_data(ROImask)
            
    tSeries = fData[ROImask.astype(bool)].T
        
    tSeries_c, ROImask_c, n = remove_broken_voxels(tSeries, ROImask)
    
    if standardize and len(tSeries_c.T)!=0:
        logging.debug("extracting_timeseries: standardizing ts")
        tSeries_c -= np.average(tSeries_c, axis=0)
        tSeries_c /= np.std(tSeries_c, axis=0)

    if sigma is not None and sigma>0 and len(tSeries_c.T)!=0:
        logging.debug("extracting_timeseries: smoothing ts")
        stSeries_c = [ sp.ndimage.gaussian_filter1d(tSeries_c[:,j], sigma=sigma)
                   for j in range(np.shape(tSeries_c)[1])]
        stSeries_c = np.asarray(stSeries_c).transpose()
    else:
        stSeries_c = tSeries_c
        
    return stSeries_c, ROImask_c, n
    



def ts_stats(tSeries):
    """
        Estimate average signal, average standard deviation and
        average SNR of a set of signals
        
        Parameters
        ----------
        tSeries     :   TxN timeseries set (T = n.timepoints, 
                                            N = n.samples)
        
        Returns
        ---------
        ts_m        :   average signal
        ts_s        :   average std dev
        SNR         :   average signal-noise ratio
        
    """
    if np.any(np.shape(tSeries))==0:
        raise ValueError("invalid shape for argument tSeries")
    if np.size(tSeries)==0:
        raise ValueError("argument tSeries is empty!")

    ts_m = np.average(tSeries, axis=1)
    ts_s = np.std(tSeries, axis=1)
    ts_s[ts_s==0] = np.nan
    if np.all(ts_s != ts_s):
        SNR = np.nan
    else:
        SNR = np.nanmean(np.abs(ts_m/ts_s))
    
    return ts_m, ts_s, SNR



def plot_meanTs(tSeries, ax=None, TR = 1, shadeColor = 'white', **plt_kwargs):
    """
        Plot average timeseries of a signals set
        
        Parameters
        ----------
        tSeries         :   TxN timeseries set (T = n.timepoints,
                                                N = n.samples)
        ax              :   matplotlib axes to use (defult = None)
        TR              :   sampling period (to scale x-axis, default = 1) 
        shadeColor      :   1-std deviation interval color

        Returns
        ----------
        SNR             :   average signal-noise ratio
        
    """
    
    if ax is None:
        logging.debug("plot_meanTs: creating new axes")
        ax = plt.gca()

    ts_m, ts_s, SNR = ts_stats(tSeries)
    
    ax.plot(np.arange(len(ts_m))*TR, ts_m, '-', **plt_kwargs)
    ax.fill_between(np.arange(len(ts_m))*TR,
                   (ts_m-ts_s), (ts_m+ts_s), color=shadeColor, alpha=.4)
    center = np.mean(ts_m)
    width = np.std(ts_m)
    ax.set_ylim([center-3.5*width,center+3.5*width])
    ax.set_xlabel("Time (s)")
    ax.set_title(f"avg SNR: {SNR:.5f}")
    return SNR


