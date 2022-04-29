# rsDPI - rs-Data Parcellation Improvement algorithm

rsDPI is a Python implementation of an algorithm to automatically refine an existing parcellation of rs-fMRI data for functional-connectivity analyses.

After cloning this reposiroty rsDPI can be run as a command line application without any installation needed.

* [Overview](#overview)
* [Requirements](#requirements)
* [Usage](#usage)
* [Author](#author)

## Overview

The algorithm addresses the problem of improving an existing parcellation of brain functional rs-fMRI data (either a whole brain parcellation or single ROIs).
In the context of brain connectome study, many experimental approaches rely on a *a priori* definition of parcellation of brain functional regions in a common standard space, to which individual data is usually registered after acquisition (MNI space).
For a number of reasons (mainly registration algorithms margins of accuracy and intrinsic individual variability of data) the definition of such a segmentation has some degree of approximation, and then could be improved by a subject-specific *fine correction*.

rsDPI algorithm basic operational principles can be summarized as:
- Extract data from pre-existing ROI (in the shape of *n* 1D temporal series)
- Compute internal functional-correlation of the ROI and select least correlated timeseries
- Remove corresponing voxels 



    