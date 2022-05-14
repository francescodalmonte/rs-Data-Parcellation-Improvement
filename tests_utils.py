# -*- coding: utf-8 -*-
import random
import pytest

import numpy as np
from hypothesis import given, assume, settings
import hypothesis.strategies as st
import hypothesis.extra.numpy as hynp

from utils import *

# random mask generator
def _rand_mask(N=100, mshape=[10,10,10]):
    random.seed(1)
    mask = np.random.rand(mshape[0],mshape[1],mshape[2])
    val = np.sort(np.reshape(mask, [-1]))[N]
    mask = mask < val    
    return mask.astype(int)



# BACK_PROJECTION()
    


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



