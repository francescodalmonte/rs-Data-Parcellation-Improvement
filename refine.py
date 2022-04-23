# -*- coding: utf-8 -*-

from utils import back_project
from nilearn import image
import numpy as np
import scipy as sp
import os



def average_correlation(tSeries):
    """
        Computes full internal correlation matrix of a ROI
        and returns average correlation values (per voxel)
        
        Parameters
        ----------
        tSereies    :   voxels timeseries
        
        Returns
        ----------
        avg_corr    :   1D array of avg correlation values
    """
        
    # internal correlation matrix   
    corrMat = np.corrcoef(tSeries, rowvar=False)
    np.fill_diagonal(corrMat,0)
    # avg corrcoef values
    avg_corr = np.mean(corrMat, axis=0)
    avg_corr[avg_corr<0] = 0
        
    return avg_corr
    


def quantile_threshold(map3D, quantileTh, onlyEdges=True):
    """
        Applies a quantile thresholding on the values of a 3D array
        
        Parameters
        ----------
        map3D       :   3D array
        quantileTh  :   fraction of lowest values to be removed
        onlyEdges   :   if "True" the function will only affect 
                        values recognized as edges
        
        Returns
        ----------
        mask_high   :   binary mask of over-threshold values
        mask_low    :   binary mask of under-threshold values
    """
    
    threshold_value = np.quantile(np.unique(map3D[map3D>0]), quantileTh)
    threshold = np.ones_like(map3D)*threshold_value
    
    mask = (map3D!=0).astype(int)
    
    if onlyEdges:
        edges = np.logical_xor(mask, sp.ndimage.binary_erosion(mask))
        mask_low = np.logical_and(map3D<threshold, edges)
    else: 
        mask_low = map3D<threshold
        
    mask_high = mask*np.logical_not(mask_low)
    
    return mask_high, mask_low




def refine_roi(tSeries, ROImask, onlyEdges = True, quantileTh = 0.25, return_mode = 'over'):
    """
        Algorithm which refines a ROI segmentation by excluding 
        least internally correlated voxels.
        
        Parameters
        ----------
        tSeries         :   timeseries set
        ROImask         :   ROI mask (numpy array or filepath)
        onlyEdges       :   if True the algorithm will eventually modify
                            only the voxels which compose the edges of
                            the ROI (default = True).
        quantileTh      :   portion of voxels to be discarded (0<quantileTh<1)
        return_mode     :   if 'over' returns mask of over-threshold voxels,
                            if 'under' returns under-threshold voxels (discarded),
                            if 'both' returns a list with both of them.
        
        Returns
        ----------
        ROImask_t       :   final ROI mask (or list of masks)
        corrMap         :   internal correlation map of the ROI (numpy array)

        Notes:
        The algorithm follows 
        a conservative approach by default (i.e. only excludes voxels 
        from the edges of the ROI), which can be changed through the 
        argument "onlyEdges".
    """    

    # check ROImask argument type (array or filepath)
    if not isinstance(ROImask, np.ndarray):
        if os.path.isfile(ROImask):
            ROImask = image.get_data(ROImask)
        else: raise TypeError("'ROImask' argument type not valid")
        
    # compute average correlations between voxels' timeseries
    avg_corr = average_correlation(tSeries)
    
    # back-project average corr. matrix to 3D map    
    corrMap = back_project(avg_corr, ROImask)

    # apply quantile threshold on correlation map 
    ROImask_h, ROImask_l = quantile_threshold(corrMap, quantileTh, onlyEdges = onlyEdges)

    if return_mode=='over': return ROImask_h, corrMap 
    elif return_mode=='under': return ROImask_l, corrMap 
    elif return_mode=='both': return [ROImask_h,ROImask_l], corrMap
    else: raise ValueError("invalid 'return-mode' argument: must be one of ['over','under','both']")
    