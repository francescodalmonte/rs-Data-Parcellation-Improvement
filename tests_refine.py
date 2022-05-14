# -*- coding: utf-8 -*-
import random
import pytest

import numpy as np
from hypothesis import given, assume
import hypothesis.strategies as st

from refine import *

# random arrays and masks generators
def _rand_array(dim=1, x=100, y=10, z=10, t=50):
    np.random.seed(1)    
    if dim==1:
        array = np.random.rand(x)
    elif dim==2:
        array = np.random.rand(t,x)
    elif dim==3:
        array = np.random.rand(x,y,z)
    elif dim==4:
        array = np.random.rand(x,y,z,t)
    return array

def _rand_mask(N=100, mshape=[10,10,10]):
    np.random.seed(1)
    mask = np.random.rand(mshape[0],mshape[1],mshape[2])
    val = np.sort(np.reshape(mask, [-1]))[N]
    mask = mask < val    
    return mask.astype(int)


# strategy for 2D lists (fake timeseries set)
dim2 = st.tuples(st.integers(min_value=2, max_value=25),
                 st.integers(min_value=2, max_value=25))
def _func2D(t):
    t = list(t)
    return st.lists(
            st.lists(st.floats(-1e8,1e8),
                     min_size = t[0], max_size = t[0]),
            min_size=t[1], max_size=t[1]            
            )   
stlist2D = dim2.flatmap(_func2D)

# strategy for 4D lists (fake fData)
dim4 = st.tuples(st.integers(min_value=3, max_value=10),
                st.integers(min_value=3, max_value=10),
                st.integers(min_value=3, max_value=10),
                st.integers(min_value=3, max_value=50))
def _func4D(t):
    t = list(t)
    return st.lists(
                st.lists(
                    st.lists(
                        st.lists(st.floats(-1e8,1e8),
                                 min_size = t[0], max_size = t[0]),
                        min_size = t[1], max_size = t[1]            
                        ),
                    min_size = t[2], max_size = t[2]            
                    ),
                min_size = t[3], max_size = t[3]            
                )
stlist4D = dim4.flatmap(_func4D)




# AVERAGE_CORRELATION()


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
    array = _rand_array(dim=2, x=100, t=50)
    av = average_correlation(array)
    
    array_sh = np.c_[array[:,50:], array[:,:50]]
    av_sh = average_correlation(array_sh)

    assert np.all(av[:50]-av_sh[50:] < 1e-15)
    assert np.all(av[50:]-av_sh[:50] < 1e-15)
    
    
    
    
# QUANTILE_THRESHOLD()


@given(th = st.floats(0,1))
def test_quantile_threshold_final_volumes(th):
    """given a random 3D correlation map and a quantileTh value,
    verify that quantile_threshold returns a mask with a volume 
    which is >= to initial volume * (1 - quantileTh) in the case 
    of onlyEdges=True; ~= to initial volume * (1 - quantileTh) otherwise"""
    map3D = _rand_array(dim=3, x=10, y=10, z=10)
    mask_high, mask_low = quantile_threshold(map3D, th,
                                             onlyEdges = True)
    assert np.sum(mask_high) + 1 >= np.size(map3D)*(1-th)

    map3D = _rand_array(dim=3, x=10, y=10, z=10)
    mask_high, mask_low = quantile_threshold(map3D, th,
                                             onlyEdges = False)
    assert np.sum(mask_high)-np.size(map3D)*(1-th)<1


@given(b = st.booleans(), th = st.floats(0,1))
def test_quantile_threshold_results_complementarity(b, th):
    """given a random 3D correlation map, verify that quantile_threshold
    results are two complementary 3D masks"""
    map3D = _rand_array(dim=3, x=10, y=10, z=10)
    mask_high, mask_low = quantile_threshold(map3D, th,
                                             onlyEdges = b)
    assert np.all(mask_high + mask_low)




# REFINE_ROI()


def test_refine_roi_return_complementarity():
    """given a random 2D array as timeseries, verify that results
    obtained are complementary if onlyEdges==False."""
    array = _rand_array(dim=2, x=200, t=50)
    mask = _rand_mask(N=200)

    over, under, _ = refine_roi(array, mask, onlyEdges = False,
                         quantileTh = 0.5)

    assert not np.all(over + under)
    
def test_refine_roi_overth_voxels_inside_roi():
    """given a random 2D array as timeseries, verify that 
    over-threshold voxels are always inside the initial ROI."""
    array = _rand_array(dim=2, x=200, t=50)
    mask = _rand_mask(N=200)
    
    over, _, _ = refine_roi(array, mask, onlyEdges = False,
                         quantileTh = 0.5)
    over_th_voxels = np.where(over)
    
    assert np.all(mask[over_th_voxels])
    
    
def test_refine_roi_onlyEdges():
    """given a random 2D array as timeseries and a mask representing
    a cubic ROI, verify that with option "onlyEdges" refine_roi only
    modifies edges of the roi."""
    array = _rand_array(dim=2, x=6**3, t=50)
    mask = np.pad(np.ones([6,6,6]), pad_width=3, constant_values=0)
    edges = np.logical_xor(mask, sp.ndimage.binary_erosion(mask.astype(bool)))
    
    refined_mask, _, _ = refine_roi(array, mask, onlyEdges = True,
                                 quantileTh = 0.5)

    diff_mask = np.abs(mask-refined_mask)
    assert np.all(edges[np.where(diff_mask)])
    
        
    