# -*- coding: utf-8 -*-
import random
import pytest

import numpy as np
from hypothesis import given, assume
import hypothesis.strategies as st

from utils import *

# random arrays and masks generators
def _rand_array(N=100):
    np.random.seed(1)    
    array = np.random.rand(N)
    return array

def _rand_array2D(N=100,t=50):
    np.random.seed(1)    
    array = np.random.rand(t,N)
    return array

def _rand_array4D(x=10,y=10,z=10,t=50):
    np.random.seed(1)    
    array = np.random.rand(x,y,z,t)
    return array

def _rand_mask(N=100, mshape=[10,10,10]):
    random.seed(1)
    mask = np.zeros(mshape)
    i=0
    while i<N:
        x,y,z = [random.choice(range(s)) for s in mshape]
        if mask[x,y,z]==0: 
            mask[x,y,z]=1
            i+=1
    return mask

# strategy for 2D lists (fake timeseries set)
dim1 = st.integers(2, 20)
list2D = dim1.flatmap(
                lambda n: st.lists(
                            st.lists(
                                st.floats(-1e8, 1e8),
                                min_size=n, max_size=n
                                )
                            )
                )



        
# BACK_PROJECTION()
    
def test_back_projection_empty_mask():
    """given an empty array, verify that back_projection
    returns an empty map"""
    
    array = [] 
    mask = _rand_mask(N=0) 
    map3D = back_project(array,mask)
    assert np.all(map3D==0)
    
@given(i1 = st.integers(0, 1000),
       i2 = st.integers(0, 1000))
def test_back_projection_different_sizes(i1,i2):
    """given array and mask with incompatible sizes, veritfy
    that back_projection raises ValueError"""
    assume(i1 != i2)
    array = _rand_array(N=i1)
    mask = _rand_mask(N=i2)
    with pytest.raises(ValueError):
        map3D = back_project(array,mask)    
    
@given(st.lists(st.floats()))
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
        remove_broken_voxels([[1],[2],[3]])

def test_remove_broken_voxels_empty_ts():
    """given an empty or invalid argument, verify that 
    broken_voxels raises the correct errors"""
    with pytest.raises(ValueError):
        remove_broken_voxels([])
    with pytest.raises(ValueError):
        remove_broken_voxels([[],[],[]])
    with pytest.raises(ValueError):
        remove_broken_voxels([[1],[2],[3]])
     
@given(list2D = list2D.filter(lambda t: len(t)>1),
       th = st.floats(min_value=1e-18, max_value=1e-2))
def test_remove_broken_voxels_PROPERTY_CONSERVATION(list2D, th):
    """given a random 2D array, verify that the sum of the discrete
    derivatives in the original array (derivatives computed along axis
    1, i.e. "time" axis) is equal to the sum of the derivatives in the
    cleaned array.
    This test proves that remove_broken_voxels only acts on signals
    (i.e. columns) with null time derivative (constant).
    NOTE:   sums are performed in two step: [np.sum(np.sum())] in both 
            cases [list2D, list2D_r] to make them numerically comparable"""
    
    list2D = np.asarray(list2D)
    list2D_r, n = remove_broken_voxels(list2D, threshold = th)
    
    z = np.zeros([1,list2D.shape[1]])
    list2D_pad1 = np.r_[list2D, z]
    list2D_pad2 = np.r_[z, list2D]
    z_r = np.zeros([1,list2D_r.shape[1]])
    list2D_r_pad1 = np.r_[list2D_r, z_r]
    list2D_r_pad2 = np.r_[z_r, list2D_r]
    
    der = np.abs(list2D_pad1 - list2D_pad2)[1:-1]
    der_r = np.abs(list2D_r_pad1 - list2D_r_pad2)[1:-1]
    
    sum_1 = np.nansum(np.sum(der, axis=0))
    sum_2 = np.sum(np.sum(der_r, axis=0))
    diff = np.abs(sum_1-sum_2)
    
    assert diff <= th*np.size(list2D)*n

# EXTRACT_TIMESERIES()

@given(b = st.booleans(), i = st.integers(0, 0))
def test_extract_timeseries_zero_fData(b,i):
    """given null fData (==0 everywhere), verify that extract_ts
    returns an empty timeseries set and remove 100% of the voxels"""
    fData = np.zeros_like(_rand_array4D())
    ROImask = _rand_mask()
    ts, mask, n = extract_timeseries(fData, ROImask, 
                                  standardize=b, sigma=i)
    assert np.size(ts)==0 and np.sum(mask)==0 and n==1
    
@given(b = st.booleans(), i = st.integers(0, 10))
def test_extract_timeseries_finalTS_number(b,i):
    """given random fData, verify that extract_ts returns a set of
    timeseries with equal or fewer samples with respect to the initial
    one (fewer only in case n - i.e. fraction of excluded voxels - > 0)"""
    fData = _rand_array4D()
    ROImask = _rand_mask()
    ts, mask, n = extract_timeseries(fData, ROImask, 
                                  standardize=b, sigma=i)

    assert len(ts.T)<=np.sum(ROImask) and np.sum(mask)<=np.sum(ROImask)
    if n>0:
        assert len(ts.T)<np.sum(ROImask) and np.sum(mask)<np.sum(ROImask)

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


@given(list2D = list2D.filter(lambda t: len(t)>1))
def test_ts_stats_tsm_bounds(list2D):
    """"given a random timeseries set, verify that ts_stats
    result ts_m maximum (and min) value is <= (>=) than max (min)
    value of original array"""
    ts_m, ts_s, SNR = ts_stats(list2D)
    assert np.max(ts_m)<=np.max(list2D)
    assert np.min(ts_m)>=np.min(list2D)    