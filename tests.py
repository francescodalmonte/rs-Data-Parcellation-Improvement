# -*- coding: utf-8 -*-
import random
import pytest

import numpy as np
from hypothesis import given, assume, settings
import hypothesis.strategies as st
import hypothesis.extra.numpy as hynp

from utils import *
from refine import *



# random mask generator
def _rand_mask(N=100, mshape=[10,10,10]):
    """
    Create a pseudo-random 3D binary mask by randomly filling 
    N points of an empty (zeros) array of shape = mshape with ones.

    Parameters
    ----------
    N        : Volume of the mask (Default = 100)
    mshape   : Shape (Default = [10,10,10])

    Returns
    ----------
    mask    : 3D mask

    """
    random.seed(1)
    mask = np.random.rand(mshape[0],mshape[1],mshape[2])
    val = np.sort(np.reshape(mask, [-1]))[N]
    mask = mask < val    
    return mask.astype(int)



# utils.py



# BACK_PROJECTION()
    


def test_back_projection_unit():
    """given an array, given a 3D mask, verify that back_project 
    returns the expected result"""
    array = [0,1,2,3,4,5,6,7,8,9] 
    mask = [[[0,0,1],[0,0,1],[0,1,0]],
            [[0,0,0],[0,0,0],[0,0,1]],
            [[0,1,1],[0,1,1],[0,1,1]]] 
    map3D = back_project(array,mask)
    
    expected = [[[np.nan,np.nan,     0],[np.nan,np.nan,     1],[np.nan,     2,np.nan]],
                [[np.nan,np.nan,np.nan],[np.nan,np.nan,np.nan],[np.nan,np.nan,     3]],
                [[np.nan,     4,     5],[np.nan,     6,     7],[np.nan,     8,     9]]]
    comparison = np.logical_xor(np.isnan(map3D), map3D == expected)
    
    assert np.all(comparison) 
    
    

def test_back_projection_empty_mask():
    """given an empty array, verify that back_projection
    returns an empty map"""
    
    array = [] 
    mask = _rand_mask(N=0) 
    map3D = back_project(array,mask)
    assert np.all(np.isnan(map3D))
    
    
    
@given(i = st.integers(0, 1000),
       array = hynp.arrays(dtype=float,
                           shape=hynp.array_shapes(min_dims=1,
                                                   max_dims=1,
                                                   min_side=0,
                                                   max_side=1000),
                           elements=st.floats(-1e8,1e8)))
def test_back_projection_different_sizes(i,array):
    """given array and mask with incompatible sizes, veritfy
    that back_projection raises ValueError"""
    assume(len(array) != i)
    mask = _rand_mask(N=i, mshape=[20,20,20])
    with pytest.raises(ValueError):
        map3D = back_project(array,mask)    
    


@given(st.lists(st.floats(1e-8,1e8)))
def test_back_projection_PROPERTY_INVERSE(array):
    """given an array, given a mask, verify that 
    back_projection returns the expected result"""
    array = np.asarray(array)
    
    mask = _rand_mask(N=len(array))
    map3D = back_project(array,mask)
    array_i = map3D[mask.astype(bool)]
        
    # if array is not empty, remove NaNs before comparison
    if len(array)>0:
        array = array[np.logical_not(np.isnan(array))]
    if len(array_i)>0:
        array_i = array_i[np.logical_not(np.isnan(array_i))]

    assert np.all(array_i == array)



# REMOVE_BROKEN_VOXELS()



def test_remove_broken_voxels_unit():
    """given a 2D array, verify that remove_broken_voxels returns 
    the expected result"""
    array2D = np.array([[1,2,1,3],
                        [1,2,4,3],
                        [1,np.nan,1,0],
                        [1,np.nan,1,3]])
    cleaned, _, n = remove_broken_voxels(array2D)
    expected = np.array([[1,3],
                         [4,3],
                         [1,0],
                         [1,3]])
    assert np.all(cleaned==expected)
    assert n == 0.5
    
    
    
def test_remove_broken_voxels_empty_ts():
    """given an empty or invalid argument, verify that 
    broken_voxels raises the correct errors"""
    with pytest.raises(ValueError):
        remove_broken_voxels([])
    with pytest.raises(ValueError):
        remove_broken_voxels([[],[],[]])
    with pytest.raises(ValueError):
        remove_broken_voxels([[1,2,3]])
        

    
@given(th = st.floats(min_value=1e-18, max_value=1e-2),
       array2D = hynp.arrays(dtype=float,
                           shape=hynp.array_shapes(min_dims=2,
                                                   max_dims=2,
                                                   min_side=2,
                                                   max_side=1000),
                           elements=st.floats(-1e8,1e8)))
def test_remove_broken_voxels_PROPERTY_CONSERVATION(th, array2D):
    """given a random 2D array, verify that the sum of the discrete
    derivatives in the original array (derivatives computed along axis
    1, i.e. "time" axis) is equal to the sum of the derivatives in the
    cleaned array.
    This test proves that remove_broken_voxels only acts on signals
    (i.e. columns) with null time derivative (constant).
    NOTE:   sums are performed in two step: [np.sum(np.sum())] in both 
            cases [array2D, array2D_r] to make them numerically comparable"""
    
    array2D_r, _, n = remove_broken_voxels(array2D, threshold = th)
    
    z = np.zeros([1,array2D.shape[1]])
    array2D_pad1 = np.r_[array2D, z]
    array2D_pad2 = np.r_[z, array2D]
    z_r = np.zeros([1,array2D_r.shape[1]])
    array2D_r_pad1 = np.r_[array2D_r, z_r]
    array2D_r_pad2 = np.r_[z_r, array2D_r]
    
    der = np.abs(array2D_pad1 - array2D_pad2)[1:-1]
    der_r = np.abs(array2D_r_pad1 - array2D_r_pad2)[1:-1]
    
    sum_1 = np.nansum(np.sum(der, axis=0))
    sum_2 = np.sum(np.sum(der_r, axis=0))
    diff = np.abs(sum_1-sum_2)
    
    assert diff <= th*np.size(array2D)*n
    
    

# EXTRACT_TIMESERIES()



def test_extract_timeseries_unit():
    """given a 4D array and a 3D binary mask, verify that
    extract_timeseries returns the expected result"""
    array4D = np.array([[[[1,2,3],[4,5,6]],
                         [[2,3,4],[5,6,7]],
                         [[3,4,5],[6,7,8]]],
                        [[[3,2,1],[6,5,4]],
                         [[4,3,2],[7,6,5]],
                         [[5,4,3],[8,7,6]]],
                        [[[9,8,7],[6,5,4]],
                         [[3,2,1],[9,8,7]],
                         [[6,5,4],[3,2,1]]]])
    mask = np.array([[[1,1],[0,0],[0,0]],
                     [[0,0],[1,0],[0,0]],
                     [[0,0],[0,1],[0,1]]])
    expected = np.array([[1,4,4,9,3],
                         [2,5,3,8,2],
                         [3,6,2,7,1]])
    extracted,_,_ = extract_timeseries(array4D, mask,
                                       standardize=False, sigma=None)
    assert np.all(extracted == expected)


@given(b = st.booleans(), i = st.floats(0, 10))
def test_extract_timeseries_zero_fData(b,i):
    """given null fData (==0 everywhere), verify that extract_ts
    returns an empty timeseries set and remove 100% of the voxels"""
    fData = np.zeros([10,10,10,100])
    ROImask = _rand_mask()
    ts, mask, n = extract_timeseries(fData, ROImask, 
                                  standardize=b, sigma=i)
    assert np.size(ts)==0 and np.nansum(mask)==0 and n==1
    
    

@given(b = st.booleans(), i = st.integers(0, 10), N = st.integers(1, 100),
       array4D = hynp.arrays(dtype=float,
                           shape=hynp.array_shapes(min_dims=4,
                                                   max_dims=4,
                                                   min_side=10,
                                                   max_side=50),
                           elements=st.floats(-1e8,1e8)))
def test_extract_timeseries_finalTS_number(b, i, N, array4D):
    """given random fData, verify that extract_ts returns a set of
    timeseries with equal or fewer samples with respect to the initial
    one (fewer only in case n - i.e. fraction of excluded voxels - > 0)"""
    ROImask = _rand_mask(N=N, mshape=np.shape(array4D)[:-1])
    ts, mask, n = extract_timeseries(array4D, ROImask, 
                                     standardize=b, sigma=i)

    assert len(ts.T)<=np.nansum(ROImask) and np.nansum(mask)<=np.nansum(ROImask)
    if n>0:
        assert len(ts.T)<np.nansum(ROImask) and np.nansum(mask)<np.nansum(ROImask)
    
    

# TS_STATS()



def test_ts_stats_unit():
    """given a 2D array, verify that ts_stats returns the expected results"""
    array2D = np.array([[1,1,1,2],
                        [1,2,1,2],
                        [2,2,3,2],
                        [2,2,2,np.nan]])
    ts_m_exp = np.array([1.25, 1.5 , 2.25, np.nan])
    ts_s_exp = np.array([0.4330127, 0.5, 0.4330127, np.nan])
    SNR_exp = 3.6943012562182536
    ts_m, ts_s, SNR = ts_stats(array2D)
    
    assert np.all(np.logical_xor(np.isnan(ts_m_exp), np.isclose(ts_m, ts_m_exp)))
    assert np.all(np.logical_xor(np.isnan(ts_s_exp), np.isclose(ts_s, ts_s_exp)))
    assert np.isclose(SNR, SNR_exp)
    
    
    
def test_ts_stats_empty_ts():
    """given invalid arguments, verify that ts_stats returns
    expected errors"""
    ts_1 = []
    with pytest.raises(ValueError):
        ts_stats(ts_1)
    ts_2 = [[],[],[]]
    with pytest.raises(ValueError):
        ts_stats(ts_2)
    ts_3 = [1,2,3,4,5]
    with pytest.raises(ValueError):
        ts_stats(ts_3)    



@given(array2D = hynp.arrays(dtype=float,
                    shape=hynp.array_shapes(min_dims=2,
                                            max_dims=2,
                                            min_side=2,
                                            max_side=20),
                    elements=st.floats(-1e8,1e8)))
def test_ts_stats_tsm_bounds(array2D):
    """"given a random timeseries set, verify that ts_stats
    result ts_m maximum (and min) value is <= (>=) than max (min)
    value of original array"""
    ts_m, ts_s, SNR = ts_stats(array2D)
    assert np.max(ts_m)<=np.max(array2D)+np.abs(np.max(array2D))*1e-12
    assert np.min(ts_m)>=np.min(array2D)-np.abs(np.min(array2D))*1e-12



# refine.py



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



