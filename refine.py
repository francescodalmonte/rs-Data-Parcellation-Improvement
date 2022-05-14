# -*- coding: utf-8 -*-

import os
import logging
import warnings

import utils
                    
from nilearn import image
import numpy as np
import scipy as sp




def average_correlation(tSeries):
    """
        Computes full internal correlation matrix of a ROI
        and returns average correlation values (per voxel)
        
        Parameters
        ----------
        tSeries    :    (array-like)
                        TxN timeseries set (T = n.timepoints,
                                            N = n. samples)
        
        Returns
        ----------
        avg_corr    :   1D array of avg correlation values
        
        
        Notes
        -----------
        This function computes a full adjacency matrix of the 
        timeseries set, using Pearson's correlation as metric
        (which is the standard accepted metric for functional brain
        connectivity measurements).
        Diagonal elements of the matrix are set to 0 before computing
        rows-averages and negative values are finally set to zero.
        Make sure you run remove_broken_voxels on tSeries before
        this function, in order to remove broken signals.
        
        Examples
        -----------
        >>> import numpy as np
        >>> from refine import average_correlation
        >>> ts = np.array([[1.,0.,2.,3.,4.,1.],
                           [5.,8.,6.,5.,7.,4.],
                           [5.,0.,6.,5.,7.,4.],
                           [4.,8.,3.,8.,2.,2.]])
        >>> average_correlation(ts)
        [Out]
        array([0.547, 0.209, 0.488, 0.161, 0.247, 0.508])
        
        
        >>> import numpy as np
        >>> from refine import average_correlation
        >>> ts = np.array([[1.,1.,1.,4.],
                           [2.,2.,2.,3.],
                           [3.,4.,4.,2.],
                           [5.,5.,5.,1.]])
        >>> average_correlation(ts)
        [Out]
        array([0.235, 0.243, 0.243, 0.])
        
    """
    
    # check tSeries arg
    utils._check_tSeries_arg(tSeries)
    
    # internal correlation matrix   
    tSeries, _, _ = utils.remove_broken_voxels(tSeries)
    utils._check_tSeries_arg(tSeries) 
    corrMat = np.corrcoef(tSeries, rowvar=False)
        
    np.fill_diagonal(corrMat,0)
    # avg corrcoef values
    avg_corr = np.mean(corrMat, axis=0)
    avg_corr[avg_corr<0] = 0
        
    return avg_corr
    


def quantile_threshold(map3D, quantileTh, onlyEdges=True):
    """
        Applies a quantile thresholding on a 3D array
        
        Parameters
        ----------
        map3D       :   (array-like)
                        3D array
                        
        quantileTh  :   (float >= 0 && <=1)
                        fraction of lowest values to be removed
                        
        onlyEdges   :   (bool)
                        if "True" the function will only affect 
                        values recognized as edges
        
        Returns
        ----------
        mask_high   :   binary mask of over-threshold values
        mask_low    :   binary mask of under-threshold values


        Notes
        ----------
        This function takes a set of values arranged in a 3D array as
        input and performs a threshold such that a fraction = quantileTh 
        of the initial values is preserved; it returns both a mask of
        over-threshold values and under-threshold values.
        NaN values in the initial array are ignored: the function initially
        creates a mask of non-NaN values and perform following steps
        only on masked values.
        
        If onlyEdges = True, the algorithm detects "edges" in the
        initial 3D map exploiting a morphological erosion of the non-NaN
        mask, and only perform the threshold for those points belonging to
        the edges (i.e. "internal" points are always included in over-th mask);
        notice that in any case the thresholding value is computed as a 
        quantile of the whole set of non-NaN values, even if onlyEdges = True,
        then in this case it is not possible to knwow what fraction of points
        will be in over- and under-threshold masks in advance.
    
        In real-world cases, the option onlyEdges = True correspond to 
        a more conservative approach.
        
        Examples
        ----------
        >>> import numpy as np
        >>> from refine import quantile_threshold
        >>> m = np.array([[[np.nan,np.nan],[np.nan,0],[2,5]],
                          [[np.nan,np.nan],[0,1],[2,4]],
                          [[np.nan,np.nan],[1,3],[6,4]]])
        >>> mask_1, mask_2 = quantile_threshold(m, 0.5)
        >>> mask_1
        [Out]
        array([[[0, 0],
                [0, 0],
                [1, 1]],

               [[0, 0],
                [0, 0],
                [1, 1]],

               [[0, 0],
                [0, 1],
                [1, 1]]])
        
        
    """
    
    threshold_value = np.quantile(map3D[map3D==map3D], quantileTh)
    if threshold_value == 0:
        warnings.warn("Using a threshold value = 0; consider increasing quantileTh value") 
    threshold = np.ones_like(map3D)*threshold_value
    
    mask = (map3D==map3D).astype(int)
    
    if onlyEdges:
        edges = np.logical_xor(mask, sp.ndimage.binary_erosion(mask))
        mask_low = np.logical_and(map3D<threshold, edges)
    else: 
        mask_low = map3D<threshold

    mask_high = mask*np.logical_not(mask_low)
    
    return mask_high, mask_low.astype(int)




def refine_roi(tSeries, ROImask, onlyEdges = True, quantileTh = 0.25, return_mode = 'over'):
    """
        Algorithm which refines a ROI segmentation by excluding 
        least internally correlated voxels.
        
        Parameters
        ----------
        tSeries         :   (array-like)
                            TxN timeseries set (T = n. timepoints
                                                N = n. samples)
                            
        ROImask         :   (str or array-like)
                            ROI mask (array or filepath)
                            
        onlyEdges       :   (bool)
                            if True the algorithm only modifies the
                            voxels of the edges of the ROI (default = True).
                            
        quantileTh      :   (float >=0 && <=1)
                            portion of voxels to be discarded
                            
        return_mode     :   {'over','under','both'}
                            if 'over' returns mask of over-threshold voxels,
                            if 'under' returns under-threshold voxels,
                            if 'both' returns a list with both of them.
        
        Returns
        ----------
        ROImask_t       :   final ROI mask (or list of masks)
        corrMap         :   internal correlation map of the ROI (numpy array)


        Notes
        ----------
        This function takes a timeseries set and the corresponding ROImask
        as inputs and produce a "refined" version of the ROI by removing
        the least correlated voxels (implicitly assuming that tSeries set
        were directly extracted with extract_timeseries() using the same
        ROImask).
        The algorithm follows a conservative approach by default (i.e.
        only excludes voxels from the edges of the ROI), which can be
        changed through the argument "onlyEdges".
        
        The function initially computes each voxel's average correlation
        with the others, than back_project those values into the original
        3D mask, and finally perform a threshold on this mask.
        
        
        See Also
        -----------
        average_correlation()
        utils.back_project()
        quantile_threshold()
    

        
    """    

    # check ROImask argument type (array or filepath)
    ROImask = utils._if_notArray_getFromFile(ROImask)
        
    # compute average correlations between voxels' timeseries
    avg_corr = average_correlation(tSeries)
    
    # back-project average corr. matrix to 3D map    
    corrMap = utils.back_project(avg_corr, ROImask)

    # apply quantile threshold on correlation map 
    logging.debug(f"Applying quantile threshold = {quantileTh} (modality '{return_mode}')")
    
    ROImask_h, ROImask_l = quantile_threshold(corrMap, quantileTh, onlyEdges = onlyEdges)

    logging.debug(f"ROI's volume (before/after): {np.sum(ROImask)}/{np.sum(ROImask_h)}")
    logging.debug(f"removed {100*(np.sum(ROImask_l))/np.sum(ROImask):.3}% volume")
    if return_mode=='over': return ROImask_h, corrMap 
    elif return_mode=='under': return ROImask_l, corrMap 
    elif return_mode=='both': return [ROImask_h,ROImask_l], corrMap
    else: raise ValueError("invalid 'return-mode' argument: must be one of ['over','under','both']")
    