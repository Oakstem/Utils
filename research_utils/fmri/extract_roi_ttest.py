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
from research_utils.fmri.glm_nnd import build_design_matrix, build_contrast_noface_events

class Dataset:
    def __init__(self, config, roi_decoder):
        self.config = config
        self.roi_decoder = roi_decoder
        self._dataset_read()
        self._prepare_results_dir(self.config['fmri_path'])

    def _dataset_read(self):
        self.fmri_path = self.config['fmri_path']
        self.func = nibabel.load(self.fmri_path)
        self.func_shape = self.func.shape
        self.load_annots(self.config['labels_path'])
        self._load_hp_movement()

    def _load_hp_movement(self):
        if self.config['hp_movement_path'] is not None:
            self.movement_data = np.loadtxt(self.config['hp_movement_path'])
            if self.config['use_only_6_motion_regressors']:
                # use only the first 6 motion regressors, there are 18 in total (6 for each run, 3 runs in total)
                self.movement_data = self.movement_data[:, self.config['head_motion_indices']]
        else:
            self.movement_data = None
        # limit the fmri data to the movement data size
        if self.movement_data is not None:
            self.func = self.func.slicer[..., :self.movement_data.shape[0]]

    def load_annots(self, labels_path, labels2_path=None, feature_1='face', feature_2='no_face'):
        annots_1 = pd.read_csv(labels_path, delimiter=' ', names=['onset', 'duration'])
        annots_1['trial_type'] = feature_1
        if labels2_path is not None:
            annots_2 = pd.read_csv(labels2_path, delimiter=' ', names=['onset', 'duration'])
            annots_2['trial_type'] = feature_2
        else:
            annots_2 = build_contrast_noface_events(annots_1, self.func.shape[-1], trial_type=feature_2)
        # create the design matrix
        events = pd.concat([annots_1, annots_2], ignore_index=True)
        events = events.sort_values(by='onset')
        events = events.reset_index(drop=True)
        self.labels = events

    def find_mean(self, limit=100):
        self.mean_img = image.mean_img(self.func.slicer[..., :limit])


    def _prepare_results_dir(self, fmri_path, dataset_name='NND'):
        self.sub_id = Path(fmri_path).stem.split('_task')[0]
        self.task_name = Path(fmri_path).stem.split('_task-')[1].split('_')[0]
        self.results_dir = Path(__file__).parent / 'results' / dataset_name / self.task_name / self.sub_id
        self.results_dir.mkdir(parents=True, exist_ok=True)
        # clear the directory if not empty
        # for file in self.results_dir.glob('*'):
        #     if 'MNI152' in file.name:
        #         # skip the MNI152 template file
        #         continue
        #     file.unlink()


    def apply_ttest(self, max_th=10., onset_th=0.8):
        self.func_data = self.func.get_fdata().astype(np.float16)
        # to free up memory, we can store only the first volume sample
        self.func = self.func.slicer[..., 0]
        print("Applying t-test...")
        feature_1 = self.design_matrix['face'] > onset_th
        feature_2 = self.design_matrix['no_face'] > onset_th
        _, p_values = stats.ttest_ind(
            self.func_data[..., feature_1],
            self.func_data[..., feature_2],
            axis=-1,
        )

        # Use a log scale for p-values
        log_p_values = -np.log10(p_values)
        # NAN values to zero
        log_p_values[np.isnan(log_p_values)] = 0.0
        log_p_values[log_p_values > max_th] = 10.0

        self.log_p_values = log_p_values
        self.log_p_values_img = new_img_like(self.func, log_p_values)


    def convert_auto_labels(self):
        # convert the automatic labels to the format used in the self
        proc_labels = np.zeros(self.func.shape[-1])
        for i in range(self.labels.shape[0]):
            st_ind = int(np.round(self.labels['onset'][i]))
            end_ind = int(np.round(self.labels['onset'][i] + self.labels['duration'][i]))
            proc_labels[st_ind:end_ind] = 1
        self.labels = proc_labels.astype(np.int16)

    def _build_design_matrix(self):
        events = self.labels
        self.design_matrix = build_design_matrix(self.func, tr=1, movement_regressor=self.movement_data, events_matrix=events)


    def plot_p_values(self, cut_coords, title='p-values'):
        plotting.plot_stat_map(
            self.log_p_values_img,
            bg_img=self.mean_img,
            # threshold=0.1,
            title=title,
            cut_coords=cut_coords,
            draw_cross=False,
            # cmap='magma',
        )
        plotting.show()

    def plot_roi(self, cut_coords, title='p-values'):
        plot_roi(
            self.roi_mask_img,
            self.mean_img,
            title=title,
            cut_coords=cut_coords,
            annotate=False,
        )
        plt.show()

    def threshold_p_values(self, threshold=2.5):
        self.log_p_values[self.log_p_values < threshold] = 0
        self.log_p_values_img = image.new_img_like(self.func, self.log_p_values)


    def prepare_mask(self, mask_path, cut_coords, mask_name='FFA'):
        ffa_mask = nibabel.load(mask_path)
        # resample the image to our scale
        ffa_mask = resample_to_img(ffa_mask, self.func)

        plot_roi(
            ffa_mask,
            self.mean_img,
            cut_coords=cut_coords,
            title=mask_name,
        )
        plt.show()

    def save_results(self):
        blur, censor, ica = self.get_fmri_type()
        result_path = self.results_dir / f'{self.sub_id}_log_p_values_{blur}_{censor}_{ica}.nii.gz'
        self.log_p_values_img.to_filename(result_path)

    def dilate(self):
        bin_p_values = self.log_p_values > 0
        self.roi_mask = binary_dilation(bin_p_values)
        self.roi_mask_img = new_img_like(self.func, self.roi_mask.astype(int))

    def get_fmri_type(self):
        blur = 'no_blur' if 'no_blur' in self.fmri_path else 'blur'
        censor = 'no_censor' if 'no_censor' in self.fmri_path else 'censor'
        ica = 'ica' if 'ica' in self.fmri_path else ''
        return blur, censor, ica

    def create_roi_masks(self, min_voxels_per_roi=20, use_neurosynth=False):
        labels, _ = sp_label(self.roi_mask)
        unq_labels = np.unique(labels)
        unq_labels = unq_labels[unq_labels > 0]  # remove the background label
        print(f'Found {unq_labels.size} unique regions in the mask')
        print(f"Filtering out regions with less than {min_voxels_per_roi} voxels")
        roi_dd = {}
        for unq_label in unq_labels:
            roi_size = np.sum(labels == unq_label)
            if roi_size >= min_voxels_per_roi:
                single_roi = self.roi_mask.copy()
                single_roi[labels != unq_label] = 0
                roi_dd[unq_label] = {'mask': new_img_like(self.func, single_roi), 'size': roi_size}

                blur, censor, ica = self.get_fmri_type()

                mask_save_path = (self.results_dir /
                                  f'roi_{unq_label}_size_{roi_size}_{blur}_{censor}_{ica}.nii.gz')

                if use_neurosynth:
                    # use the neurosynth decoder to find the name of the ROI based on it's database of studies [this takes a while]
                    print(f'ROI:{unq_label}, Searching for ROI name using neurosynth...')
                    results = self.roi_decoder.get_roi_name(roi_dd[unq_label]['mask'], decoder_type='neurosynth-roi')
                    if results['top_name'] is not None:
                        mask_save_path = (self.results_dir /
                                          f'roi_size_{unq_label}_{roi_size}_name_{results["top_name"]}_'
                                          f'{roi_size}_{blur}_{censor}_{ica}.nii.gz')
                        print(f'Found ROI:{unq_label} decoded_name:{results["top_name"]}, XYZ:{results["center_coords"]}, size:{roi_size}')
                else:
                    print(f'Found ROI:{unq_label} size:{roi_size} voxels')

                roi_dd[unq_label]['mask'].to_filename(mask_save_path)
                roi_dd.pop(unq_label)





if __name__ == '__main__':
    cut_coords = [-40, -55, -10]
    p_values_th = 8
    min_voxels_per_roi = 100

    config = {
        'fmri_path': r"E:\NND\ds002837-download\derivatives\sub-1\func\sub-1_task-500daysofsummer_bold_blur_censor.nii.gz",
        'labels_path': r"E:\NND\ds002837-download\stimuli\task-500daysofsummer_face-annotation.1D",
        'hp_movement_path': None,
        'neurosynth_dataset': None,
        'use_only_6_motion_regressors': True,
        'head_motion_indices': np.arange(-36, -18),
    }

    roi_decoder = ROIDecoder(config['neurosynth_dataset'])
    # for fmri_path in glob.glob('/Users/alonz/PycharmProjects/pSTS_DB/psts_db/datasets/NND/derivatives/sub*/func/sub*_task-500daysofsummer_bold_blur_censor_ica.nii.gz'):
    # config['fmri_path'] = fmri_path
    print(f'Processing Subject: {config["fmri_path"]}')
    dataset = Dataset(config, roi_decoder)
    # dataset.convert_auto_labels()
    dataset._build_design_matrix()
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

