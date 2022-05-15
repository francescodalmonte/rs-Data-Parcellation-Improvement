# -*- coding: utf-8 -*-
import random
import pytest

import numpy as np
from hypothesis import given, assume
import hypothesis.strategies as st
import hypothesis.extra.numpy as hynp

from refine import *

# random arrays and masks generators

def _rand_mask(N=100, mshape=[10,10,10]):
    np.random.seed(1)
    mask = np.random.rand(mshape[0],mshape[1],mshape[2])
    val = np.sort(np.reshape(mask, [-1]))[N]
    mask = mask < val    
    return mask.astype(int)



# AVERAGE_CORRELATION()



def test_average_correlation_unit():
    """given a 2D array, verify average_correlation returns
    the expected result"""
    array = np.array([[7,9,9,5,9,8],
                      [3,8,8,5,8,7],
                      [4,5,7,5,5,4],
                      [1,4,5,1,3,3],
                      [1,1,2,1,2,2]])
    expected = np.array([0.68292592, 0.75797753, 0.75499549,
                         0.69981831, 0.76911863, 0.75248621])
    result = average_correlation(array)
    
    assert np.all(np.isclose(result, expected))


def test_average_correlation_removing_broken():
    """given a 2D array, verify that average_correlation returns
    an array of the expected length"""
    array_1 = np.asarray([[2,1,4],
                        [3,2,2],
                        [1,1,1]])
    assert len(average_correlation(array_1))==3    
    array_2 = np.asarray([[2,1,4],
                        [3,1,2],
                        [1,1,1]])
    assert len(average_correlation(array_2))==2
    array_3 = np.asarray([[2,np.nan,4],
                        [3,5,2],
                        [1,1,1]])
    assert len(average_correlation(array_2))==2
    
    
    
def test_average_correlation_empty_ts():
    """given an empty array or an array with invalid shape, verify 
    that average_correlation raises the correct errors"""
    with pytest.raises(ValueError):
        average_correlation([])
    with pytest.raises(ValueError):
        average_correlation([1])
    with pytest.raises(ValueError):
        average_correlation([1,2,3])
    with pytest.raises(ValueError):
        average_correlation([[1],[2],[3]])
    with pytest.raises(ValueError):
        average_correlation([[1,2,3],
                             [1,2,3]])
        
        

def test_average_correlation_shuffling_ts():
    """given a random 2D array of timeseries, and an array
    with the same timeseries sorted in a different order,
    verify that the resulting vectors contain the same values"""
    
    np.random.seed(1)
    array2D = np.random.rand(5,20)
    
    av = average_correlation(array2D)
    
    array2D_sh = np.c_[array2D[:,10:], array2D[:,:10]]
    av_sh = average_correlation(array2D_sh)

    assert np.all(av[:10]-av_sh[10:] < 1e-15)
    assert np.all(av[10:]-av_sh[:10] < 1e-15)
    
    
    
# QUANTILE_THRESHOLD()



def test_quantile_threshold_unit():
    """given a 3D array and a threshold value, verify that 
    quantile_threshold returns the expected result"""
    array = np.array([[[np.nan,np.nan],[np.nan,0],[2     ,5]],
                      [[np.nan,np.nan],[0     ,1],[2     ,4]],
                      [[np.nan,np.nan],[1     ,3],[6     ,4]],
                      [[7     ,     6],[1     ,3],[6     ,4]],
                      [[9     ,     8],[6     ,7],[2     ,1]]])
    th = 0.25
    expected_over = np.array([[[0, 0],[0, 0],[1, 1]],
                              [[0, 0],[0, 0],[1, 1]],
                              [[0, 0],[0, 1],[1, 1]],
                              [[1, 1],[0, 1],[1, 1]],
                              [[1, 1],[1, 1],[1, 0]]])
    expected_under = np.array([[[0, 0],[0, 1],[0, 0]],
                               [[0, 0],[1, 1],[0, 0]],
                               [[0, 0],[1, 0],[0, 0]],
                               [[0, 0],[1, 0],[0, 0]],
                               [[0, 0],[0, 0],[0, 1]]])

    result_over, result_under = quantile_threshold(array, th)
    assert np.all(result_over==expected_over)
    assert np.all(result_under==expected_under)


    
@given(th = st.floats(0.01,0.99),
       array3D = hynp.arrays(dtype=float,
                             shape=hynp.array_shapes(min_dims=3,
                                                     max_dims=3,
                                                     min_side=5,
                                                     max_side=20),
                             elements=st.floats(1e-8,1e8)))
def test_quantile_threshold_final_volume_1(th, array3D):
    """given a random 3D correlation map and a quantileTh value,
    verify that quantile_threshold returns a mask with a volume 
    which is >= to initial volume * (1 - quantileTh) in the case 
    of onlyEdges=True"""
    
    mask_high, mask_low = quantile_threshold(array3D, th,
                                             onlyEdges = True)
    assert np.sum(mask_high) >= np.size(array3D)*(1-th)



@given(th = st.floats(0.01,0.99),
       array3D = hynp.arrays(dtype=float,
                             shape=hynp.array_shapes(min_dims=3,
                                                     max_dims=3,
                                                     min_side=5,
                                                     max_side=20),
                             elements=st.floats(1e-8,1e8)))
def test_quantile_threshold_final_volume_2(th, array3D):
    """given a random 3D correlation map and a quantileTh value,
    verify that quantile_threshold with onlyEdges=False returns
    a mask with a volume <= than that returned with onlyEdges=True"""

    mask_high_1, mask_low_1 = quantile_threshold(array3D, th,
                                                 onlyEdges = False)
    mask_high_2, mask_low_2 = quantile_threshold(array3D, th,
                                                 onlyEdges = True)
    assert np.sum(mask_high_1)<=np.sum(mask_high_2)
    assert np.sum(mask_low_1)>=np.sum(mask_low_2)
    
    

@given(b = st.booleans(), th = st.floats(0.01,0.99),
       array3D = hynp.arrays(dtype=float,
                             shape=hynp.array_shapes(min_dims=3,
                                                     max_dims=3,
                                                     min_side=5,
                                                     max_side=20),
                             elements=st.floats(1e-8,1e8)))
def test_quantile_threshold_results_complementarity(b, th, array3D):
    """given a random 3D correlation map, verify that quantile_threshold
    results are two complementary 3D masks"""

    mask_high, mask_low = quantile_threshold(array3D, th,
                                             onlyEdges = b)
    assert np.all(mask_high + mask_low)
    

    
# REFINE_ROI()



def test_refine_roi_unit():
    """given a 2D array and a mask, verify that refine_roi
    returns the expected results"""
    array = np.array([[1,3,2,2,1,2],
                      [2,4,3,4,5,3],
                      [5,6,7,4,4,4],
                      [7,7,8,7,6,8],
                      [7,8,9,8,9,9],
                      [9,9,9,9,9,9]])
    mask = np.array([[[1,1],[0,0],[0,0]],
                     [[0,0],[1,0],[0,0]],
                     [[0,0],[1,1],[0,1]]])
    th = 0.4
    expected_over = np.array([[[1, 1],[0, 0],[0, 0]],
                              [[0, 0],[0, 0],[0, 0]],
                              [[0, 0],[1, 0],[0, 1]]])
    
    expected_under = np.array([[[0, 0],[0, 0],[0, 0]],
                               [[0, 0],[1, 0],[0, 0]],
                               [[0, 0],[0, 1],[0, 0]]])
    
    expected_corrmap = np.array([[[0.78526322, 0.79673965],
                                  [    np.nan,     np.nan],
                                  [    np.nan,     np.nan]],
                                 [[    np.nan,     np.nan],
                                  [0.76693893,     np.nan],
                                  [    np.nan,     np.nan]],
                                 [[    np.nan,     np.nan],
                                  [0.78965239, 0.74584578],
                                  [    np.nan, 0.78550939]]])

    over, under, corrmap = refine_roi(array, mask, onlyEdges = False,
                                      quantileTh = th)

    assert np.all(over == expected_over)
    assert np.all(under == expected_under)
    assert np.all(np.logical_xor(np.isnan(expected_corrmap),
                                          np.isclose(expected_corrmap, corrmap)))



@given(b = st.booleans(), th = st.floats(0.01, 0.99))
def test_refine_roi_return_complementarity(b, th):
    """given a 2D array as timeseries and a mask, verify that results
    obtained are complementary if onlyEdges==False."""
    np.random.seed(1)
    array2D = np.random.rand(10,200)
    mask = _rand_mask(N=200)

    over, under, _ = refine_roi(array2D, mask, onlyEdges = b,
                         quantileTh = th)

    assert not np.all(over + under)
    

@given(b = st.booleans(), th = st.floats(0.01, 0.99))
def test_refine_roi_overth_voxels_inside_roi(b, th):
    """given a 2D array as timeseries and a mask, verify that 
    over-threshold voxels are always inside the initial ROI."""
    np.random.seed(1)
    array2D = np.random.rand(10,200)
    mask = _rand_mask(N=200)
    
    over, _, _ = refine_roi(array2D, mask, onlyEdges = b,
                         quantileTh = th)
    over_th_voxels = np.where(over)
    
    assert np.all(mask[over_th_voxels])
    
    
@given(th = st.floats(0.01, 0.99))
def test_refine_roi_onlyEdges(th):
    """given a 2D array as timeseries and a mask representing
    a cubic ROI, verify that with option "onlyEdges" refine_roi only
    modifies edges of the roi."""
    np.random.seed(1)
    array2D = np.random.rand(10,6**3)
    mask = np.pad(np.ones([6,6,6]), pad_width=3, constant_values=0)
    mask = np.pad(np.ones([6,6,6]), pad_width=3, constant_values=0)
    edges = np.logical_xor(mask, sp.ndimage.binary_erosion(mask.astype(bool)))
    
    refined_mask, _, _ = refine_roi(array2D, mask, onlyEdges = True,
                                 quantileTh = th)

    diff_mask = np.abs(mask-refined_mask)
    assert np.all(edges[np.where(diff_mask)])
    
    

@given(b = st.booleans(), th = st.floats(0.01, 0.99))
def test_refine_roi_2thresholds_inclusion(b, th):
    """given a 2D array as timeseries and a mask, verify that refined roi
    obtained with a lower th is always included inside the one obtained
    with a higher threshold."""
    np.random.seed(1)
    array2D = np.random.rand(10,200)
    mask = _rand_mask(N=200)
        
    refined_1, _, _ = refine_roi(array2D, mask, onlyEdges = b,
                                 quantileTh = th)
    refined_2, _, _ = refine_roi(array2D, mask, onlyEdges = b,
                                 quantileTh = th+(1-th)/2)

    assert np.all(refined_1[np.where(refined_2)])
