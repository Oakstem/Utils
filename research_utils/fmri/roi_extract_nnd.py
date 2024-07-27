import glob
import nibabel
from scipy.ndimage import label as sp_label
from matplotlib import pyplot as plt
from nilearn.plotting import plot_roi
from sklearn.utils import Bunch
import pandas as pd
import numpy as np
from nilearn import image
from scipy import stats
from nilearn.image import new_img_like, resample_to_img
from nilearn import plotting
from pathlib import Path
from scipy.ndimage import binary_dilation
from roi_decoder import ROIDecoder
from extract_roi_ttest import Dataset

if __name__ == '__main__':
    cut_coords = [-40, -55, -10]
    p_values_th = 3
    min_voxels_per_roi = 30

    fmri_path = r"E:\NND\ds002837-download\derivatives\sub-1\func\sub-1_task-500daysofsummer_bold_blur_censor.nii.gz"
    labels_path = r"E:\NND\ds002837-download\stimuli\task-500daysofsummer_face-annotation.1D"
    neurosynth_dataset = None

    roi_decoder = ROIDecoder(neurosynth_dataset)
    for fmri_path in glob.glob(r'E:\NND\ds002837-download\derivatives\sub-*\func\sub*_task-500daysofsummer_bold_blur_censor_ica.nii.gz'):
        print(f'Processing Subject: {fmri_path}')
        dataset = Dataset(fmri_path, labels_path, roi_decoder)
        dataset.convert_auto_labels()
        dataset.find_mean(limit=100)
        dataset.apply_ttest()
        dataset.plot_p_values(cut_coords, title='p-values')
        dataset.threshold_p_values(threshold=p_values_th)
        dataset.plot_p_values(cut_coords, title='thresholded p-values')
        dataset.dilate()
        dataset.plot_roi(cut_coords, title='Dilated ROI Mask')
        dataset.create_roi_masks(min_voxels_per_roi=min_voxels_per_roi, use_neurosynth=True)
        dataset.save_results()
        print(f'Saved results to {dataset.results_dir}')