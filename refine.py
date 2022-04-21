# -*- coding: utf-8 -*-

from nilearn import image
import numpy as np
import scipy as sp
import os


def refine_roi(tSeries, ROImask, onlyEdges = True, quantileTh = 0.25):
    """
        Algorithm which refines a ROI segmentation by excluding 
        least internally correlated voxels. The algorithm follows 
        a conservative approach by default (i.e. only excludes voxels 
        from the edges of the ROI), which can be changed through the 
        argument "onlyEdges".
        
        ---- Parameters ----
        tSeries         :   timeseries set
        ROImask         :   ROI mask (numpy array or filepath)
        onlyEdges       :   if True the algorithm will eventually modify
                            only the voxels which compose the edges of
                            the ROI.
        quantileTh      :   portion of voxels to be discarded (0<quantileTh<1)
        
        ---- Returns ----
        
        ROImask_t       :   final ROI mask (numpy array)
        corrMap         :   internal correlation map of the ROI (numpy array)

    """    

    # internal correlation matrix   
    corrMat = np.corrcoef(tSeries, rowvar=False)
    np.fill_diagonal(corrMat,0)

    # avg corrcoef values
    avg_corrMat = np.mean(corrMat, axis=0)
    avg_corrMat[avg_corrMat<0] = 0

    # check ROImask argument type (array or filepath)
    if isinstance(ROImask, np.ndarray):
        corrMap = np.zeros_like(ROImask)
    elif os.path.isfile(ROImask):
        ROImask = image.get_data(ROImask)
        corrMap = np.zeros_like(ROImask)
    else: raise TypeError("'ROImask' argument type not valid")
    
    # back-project average corr. matrix to 3D map    
    n=0
    for x in range(np.shape(ROImask)[0]):
        for y in range(np.shape(ROImask)[1]):
            for z in range(np.shape(ROImask)[2]):
                if ROImask[x,y,z]:
                    corrMap[x,y,z] = avg_corrMat[n]
                    n+=1
    
    if onlyEdges:
        # select edges of the ROI
        edges = np.logical_xor(ROImask,
                        sp.ndimage.binary_erosion(ROImask))
    
        # quantile threshold
        threshold_value = np.quantile(avg_corrMat, quantileTh)
        threshold = np.ones_like(corrMap)*threshold_value
        cut = np.logical_and(corrMap<threshold, edges)
        ROImask_t = ROImask*np.logical_not(cut)
    
    else: 
        # select edges of the ROI
        edges = np.logical_xor(ROImask,
                        sp.ndimage.binary_erosion(ROImask))
    
        # quantile threshold
        threshold_value = np.quantile(avg_corrMat, quantileTh)
        threshold = np.ones_like(corrMap)*threshold_value
        cut = np.logical_and(corrMap<threshold, edges)
        ROImask_t = ROImask*np.logical_not(cut)
            

    return ROImask_t, corrMap 

