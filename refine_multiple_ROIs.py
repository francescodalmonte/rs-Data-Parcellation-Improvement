# -*- coding: utf-8 -*-#

from matplotlib import pyplot as plt
from nilearn import image
import time 
import os

from utils import extract_timeseries, ts_stats, plot_meanTs
from refine import refine_roi

subj = "21G12282"
fData_path = f"F:/neurocovid/HC/{subj}/filtered_func_data_clean_toMNI.nii.gz"
ROImasks_path = "C:/Users/Francesco/thesis_neurocovid/Data/ROIs_BN_DMN" 

if not os.path.isdir(f"{ROImasks_path}/{subj}"):
    os.mkdir(f"{ROImasks_path}/{subj}")
#%%
start_time = time.time()

# load data from EPI
fData = image.get_data(fData_path)

SNR = []
SNR_t = []

files = os.listdir(ROImasks_path)

for filename in files:
    if filename[:4] == 'roi_':
        singleStart = time.time()

        print(filename)
        
        ROImask = image.get_data(f"{ROImasks_path}/{filename}")
        
        # extract ROI timeseries
        print("Extracting timeseries...")
        stSeries, ROImask, _ = extract_timeseries(fData, ROImask, sigma = 2.5)
        
        # compute SNR
        _, _, s = ts_stats(stSeries)
        SNR.append(s)

        # refine roi
        print("ROI refining...")
        ROImask_t, corrMap = refine_roi(stSeries,
                                        ROImask,
                                        onlyEdges = True,
                                        quantileTh = 0.50)

        # new timeseries
        print("Extracting new timeseries...")
        stSeries_t, ROImask_t, _ = extract_timeseries(fData, ROImask_t, sigma = 2.5)

    
        # compute SNR
        _, _, s = ts_stats(stSeries_t)
        SNR_t.append(s)

        # save results to file
        img_newROI = image.new_img_like(image.load_img(f"{ROImasks_path}/{filename}"),
                                ROImask_t.astype(int)) 
        img_newROI.to_filename(f"{ROImasks_path}/{subj}/new_{filename}")
        
        fig, ax = plt.subplots(figsize=(6,2), dpi=300, tight_layout=True)
        plot_meanTs(stSeries_t, ax=ax, TR = 0.735, shadeColor = 'grey',
                    linewidth=1, color='black')
        ax.set_ylim([-2.5,-2.5])
        fig.savefig(f"{ROImasks_path}/{subj}/{filename[:6]}_ts.jpg", format="jpg")
        
        print(f"{(time.time()-singleStart):.4}s")
        
print(f"Total elapsed time {time.time()-start_time:.4}s ")

