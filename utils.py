# -*- coding: utf-8 -*-

import os
import logging
import warnings


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
        
        
        Notes
        ----------
        This function assign values of array1D to non-zero positions
        of ROImask, following incresing order of indexes. 
        It is used in refine_roi() function to re-map a 1D
        array of average correlation values of the voxels of a
        ROI (computed with average_correlation() ) onto the 3D
        original space.
        
        
        Examples
        ----------
        >>> import numpy as np
        >>> from utils import back_project
        >>> x = np.array([2,3,1,5,7])
        >>> m = np.array([[[0,0,0],[0,0,0],[1,0,0]],
                         [[0,1,1],[0,0,0],[1,0,0]],
                         [[0,0,0],[0,1,0],[0,0,0]]])
        >>> back_project(x,m)
        [Out]
        array([[[nan, nan, nan],
                [nan, nan, nan],
                [ 2., nan, nan]],

               [[nan,  3.,  1.],
                [nan, nan, nan],
                [ 5., nan, nan]],

               [[nan, nan, nan],
                [nan,  7., nan],
                [nan, nan, nan]]])
        
        See Also
        ----------
        refine_roi()
        average_correlation()
        
    """
    array1D = np.asarray(array1D)
    ROImask = np.asarray(ROImask)
    
    if not ((ROImask==0) | (ROImask==1)).all():
        raise ValueError("ROImask must be a binary mask (must contain only 0/1)")
    
    if np.sum(ROImask)!=len(array1D):
        raise ValueError("x length must match ROI size")
    map3D = np.full_like(ROImask, np.nan, dtype=float)
    
    map3D[np.nonzero(ROImask)]=array1D
    
    return map3D



def _check_tSeries_arg(tSeries):
    """
        Check size and shape of a timeseries set array.
        To be valid, timeseries sets must be stored in a 2D array
        (TxN, with T = timepoints, N = samples), each dimension
        must have len > 0, number of timepoints must be >= 1; otherwise
        it raises a ValueError.
        
        Parameters
        ----------
        tSeries     : timeseries set (array)
        
        Returns
        ----------
        None
        
        
    """
    tSeries = np.asarray(tSeries)
    if np.size(tSeries.T)==0: 
        raise ValueError("input tSeries is empty!")
        
    try: 
        np.shape(tSeries)[0]
    except: 
        raise ValueError("invalid tSeries argument ")
        
    if np.shape(tSeries)[0]<=1:
        raise ValueError("input tSeries has <=1 timepoints!")   



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
        ROImask_clean   :   cleaned mask (or NaN)
        n               :   fraction of removed samples
        
        
        Notes
        ----------
        Timeseries are considered "broken" when signal is constant 
        for its whole length (threshold value serves as margin of 
        approximation). 
        The function, for each single timeseries, computes its mean
        value and check if absolute displacement from such value 
        is always < threshold: if True, the timeseries is considered
        as broken.
        Timeseries are also considered broken if they contain any 
        NaN values.
        If ROImask is provided, the function also remove voxels
        corresponding to broken timeseries from it (exploiting 
        back_project())
        
        See Also
        ----------
        back_project()
        
        Examples
        ----------
        >>> import numpy as np
        >>> from utils import remove_broken_voxels
        >>> ts = np.array([[0.,1.,2.,1.],
                           [1.,1.,3.,2.],
                           [2.,1.,3.,np.nan],
                           [3.,1.,2.,3.],
                           [2.,1.,2.,4.],
                           [3.,1.,2.,4.]])
        >>> new_ts, n = remove_broken_voxels(ts)
        >>> new_ts
        [Out]
        array([[0., 2.],
               [1., 3.],
               [2., 3.],
               [3., 2.],
               [2., 2.],
               [3., 2.]])
        >>> n
        [Out]
        0.5
        
    """

    _check_tSeries_arg(tSeries)
    
    tSeries = np.asarray(tSeries)
    
    # voxels with const timeseries for the whole rs length
    broken_1 = np.all(np.abs(tSeries-np.mean(tSeries, axis=0))<threshold, axis=0).astype(int)
    # voxels with NaN values
    broken_2 = np.any(np.isnan(tSeries), axis=0).astype(int)
    
    # broken voxels
    broken_voxels = np.logical_or(broken_1, broken_2)
    
    if not np.any(broken_voxels):
        logging.debug(f"no broken voxels")
        return tSeries.copy(), ROImask, 0
    
    else:
        # remove corresponding ts
        tSeries_clean = tSeries[:,np.logical_not(broken_voxels)].copy()
        # fraction of broken voxels
        n = (len(tSeries.T)-len(tSeries_clean.T))/len(tSeries.T)
        logging.debug(f"% broken voxels: {n*100:.2f}")
        if ROImask is not None:
            # remove voxels from ROImask
            ROImask_clean = ROImask - back_project(broken_voxels, ROImask)
        else:
            ROImask_clean = None
            
        return tSeries_clean, ROImask_clean, n



def _if_notArray_getFromFile(x):
    """
        If argument x is an array-like object returns x, otherwise
        checks whether x is a path to an image and eventually load it
        as array. If x is neither a path, raises TypeError
        
        Parameters
        ----------
        x   : image (array) or filepath (string)
        
        Returns
        ----------
        y   : image (array)
        
    """
    if not isinstance(x, np.ndarray):
        if isinstance(x, str):
            if os.path.isfile(x):
                logging.debug(f"extracting_timeseries: argument x is path: getting data from {x}")
                y = image.get_data(x)
        else:
            raise TypeError(f"invalid type for argument x")
    return y



def extract_timeseries(fData, ROImask, sigma = None, standardize = True):
    """
        Load timeseries from EPI (filepath or object) using a mask
        
        Parameters
        ----------
        fData       :   (str or array-like object)
                        path to EPI data or 4D timeseries set
        ROImask     :   (str or array-like object)
                        path to ROI mask file or mask binary array
        sigma       :   (float)   
                        Gaussian sigma for timeseries smoothing 
                        (default=None, i.e. no smoothing)
        standardize :   (bool)
                        whether to standardize extracted signals 

        Returns
        ---------
        stSeries_c      :   TxN cleaned/smoothed ROI's timeseries set
                            (i.e. without broken voxels)
                            (T = n.timepoints, N = n.samples)                            
        ROImask_c       :   cleaned ROImask 
        n               :   fraction of broken voxels
        
        
        Notes
        ----------
        This function uses a binary ROImask to extract a
        subset of timeseries from an EPI image (rs-fMRI data).
        Functional data can be passed as path to a *.nii.gz file
        (in that case exploiting Nilear.image.get-data() ) or
        directly as a 4D array object.
        By default, it authomatically removes broken voxels from
        the timeseries set by running remove_broken_voxels(),
        and returns a "cleaned" version of the ROImask given as
        argument.
        Arguments sigma and standardize eventually control
        the basic preprocessing of the signal.
        
        See Also:
        ----------
        remove_broken_voxels()
        scipy.ndimage.gaussian_filter1d()
        
        
    """
    
    fData = _if_notArray_getFromFile(fData)
    ROImask = _if_notArray_getFromFile(ROImask)
        
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
        average SNR of a set of signals.
        
        Parameters
        ----------
        tSeries     :   (array-like)
                        TxN timeseries set (T = n.timepoints, 
                                            N = n.samples)
        
        Returns
        ---------
        ts_m        :   average signal
        ts_s        :   average std dev
        SNR         :   average signal-noise ratio
        
        Notes
        ----------
        This function computes an average signal out of
        a set of timeseries of equal length.
        For each timepoint, SNR is estimated as the absolute
        value of signal mean / signal standard deviation; average
        SNR is then estimated as average value over the whole
        signals length. 

    """
    
    _check_tSeries_arg(tSeries)

    ts_m = np.average(tSeries, axis=1)
    ts_s = np.std(tSeries, axis=1)
    ts_s[ts_s==0] = np.nan
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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
    
    _check_tSeries_arg(tSeries)
    
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

