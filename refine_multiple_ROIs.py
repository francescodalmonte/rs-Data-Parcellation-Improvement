# -*- coding: utf-8 -*-#
               
from nilearn import image
import time 
import os

from utils import extract_timeseries, tsSNR
from refine import refine_roi

subj = "18G00027"
fData_path = f"F:/neurocovid/HC/{subj}/filtered_func_data_clean_toMNI.nii"
ROImasks_path = "C:/Users/Francesco/thesis_neurocovid/Data/ROIs_BN_DMN/" 

#%%
start_time = time.time()

# load data from EPI
fData = image.get_data(fData_path)

SNR = []
SNR_new = []

files = os.listdir(ROImasks_path)

for filename in files:
    if filename[:4] == 'roi_':
        print(filename)
        
        ROImask = image.get_data(f"{ROImasks_path}/{filename}")
        
        # extract ROI timeseries
        print("part1")
        stSeries, _ = extract_timeseries(fData, ROImask, sigma = 2.5)
        
        # compute SNR
        print("part2")
        SNR.append(tsSNR(stSeries))

        # refine roi
        print("part3")
        ROImask_t, corrMap = refine_roi(stSeries,
                                        ROImask,
                                        onlyEdges = True,
                                        quantileTh = 0.50)

        # new timeseries
        print("part4")
        stSeries_new, _ = extract_timeseries(fData, ROImask_t, sigma = 2.5)

    
        SNR_new.append(tsSNR(stSeries_new))
        print("part5")
        img_newROI = image.new_img_like(image.load_img(f"{ROImasks_path}/{filename}"),
                                ROImask_t.astype(int)) 
        img_newROI.to_filename(f"{ROImasks_path}/{subj}/new_{filename}")

        

print(f"Total elapsed time {time.time()-start_time:.4}s ")

