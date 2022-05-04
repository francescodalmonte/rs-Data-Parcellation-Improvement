# rsDPI: rs-Data Parcellation Improvement algorithm


![Alt text](img/image.JPG?raw=true "Refining a ROI improves SNR")


rsDPI is a Python implementation of an algorithm to automatically refine an existing parcellation of rs-fMRI data for functional-connectivity analyses.

After cloning this reposiroty rsDPI can be run as a command line application without any installation needed.

* [Overview](#overview)
* [Requirements](#requirements)
* [Basic usage](#usage)
* [Author](#author)

## Overview

The algorithm addresses the problem of improving an existing parcellation of brain functional rs-fMRI data (either a whole brain parcellation or single ROIs).
In the context of brain connectome study, many experimental approaches rely on a *a priori* definition of parcellation of brain functional regions in a common standard space, to which individual data is usually registered after acquisition (MNI space).
For a number of reasons (mainly registration algorithms margins of accuracy and intrinsic individual variability of data) the definition of such a segmentation has some degree of approximation, and then could be improved by a subject-specific *fine correction*.

rsDPI algorithm basic operational principles can be summarized as:
- Extract data from pre-defined ROI (in the shape of n 1D temporal series)
- Compute internal functional-correlation of the ROI and select least correlated timeseries
- Remove corresponding voxels 

In practise rsDPI algorithm can be run in two different modalities: singleROI or multiROI refinement; in the first one a single ROI is given to the algorithm as input, while in the second it is possible to pass a whole directory containing many ROIs to be processed.

After running rsDPI-algorithm on a set of ROIs, results with real-world data show an increased SNR of the mean-timeseries of each ROI, and an increased functional connectivity between different ROIs.

## Requirements

This application was tested on Python 3.8.5, running on Windows OS.
Minimal required Python libraries are:
- `Numpy`
- `Scipy`
- `Matplotlib`
- `Nilearn`

## Basic usage 

To run the algorithm on your machine clone this repository with:
```
git clone https://github.com/francesco28597/rs-Data-Parcellation-Improvement
cd rs-Data-Parcellation-Improvement
```
Then you'll have to set up the configuration file `/config.INI` which by default looks like this:
```
[singleMode]
fData_path = path/to/func_data.nii.gz
ROI_path = path/to/ROI_mask.nii.gz
newROI_dirpath = path/to/newROI_mask/directory
results_dirpath = path/to/results/directory
sigma = 2.5
qTh = 0.25

[multiMode]
fData_path = path/to/func_data.nii.gz
ROI_dirpath = path/to/ROIs_masks/directory
newROI_dirpath = path/to/newROIs_masks/directory
results_dirpath = path/to/results/directory
sigma = 2.5
qTh = 0.25
```
Depending on the modality you will use, you may only configure one of the two sections.
`fData_path` should be the absolute path to a *.nii.gz EPI image (from a rs-fMRI experiment). `ROI_path`/`ROI_dirpath` are absolute paths to the original ROI/the original directory containing the set of ROIs you want to refine. `newROI_dirpath`/`results_dirpath` are absolute paths to results directories (can be equal)
Finally `sigma` and `qTh` are parameters to be passed to the algorithm; respectively they represent the amplitude of the smoothing we want to apply on the original timeseries (pre-processing step), and can be set to `0` if you want no smoothing, and the *quantile threshold* value for the refinement (i.e. an upper bound for the fraction of voxels to be discarded from each ROI).

Algorithm can be launched with the following command in singleROI mode (in this case using `verbose` flag)
```
python rsDPI.py -v
```
or in multiROI mode
```
python rsDPI.py -m -v
```

In /`rois` are available a set of MNI-registered ROIs from the Brainnetome DMN parcellation (atlas.brainnetome.org), which can be used to test the algorithm; in /`std` are available some standard T1-brains from MNI, useful for visualization of results. If you don't need any of those you may simply delete the two directories.


## Author

Francesco Dalmonte

MSc student at University of Bologna