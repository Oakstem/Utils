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
from nilearn.plotting import plot_design_matrix
from nilearn.image import concat_imgs, mean_img, resample_img
from nilearn.glm.first_level import make_first_level_design_matrix, FirstLevelModel


def build_design_matrix(fmri_data, tr=1, movement_regressor=None, events_matrix=None,
                        hrf_model="glover", drift_model="polynomial", drift_order=1):
    """
    Function to build the design matrix for the GLM model
    :param fmri_data: the fmri data
    :param tr: the repetition time
    :param movement_regressor: head movement regressor indicators
    :param events_matrix: events dataframe with columns: onset, duration, trial_type
    """
    if movement_regressor is not None:
        # add head movement regressor names
        add_reg_names = [f"regr_{ind}" for ind in range(movement_regressor.shape[1])]
        movement_sz = movement_regressor.shape[0]
    else:
        add_reg_names = None
        movement_sz = fmri_data.shape[-1]

    n_scans = fmri_data.shape[-1]
    frame_times = np.arange(n_scans) * tr
    design_mat = make_first_level_design_matrix(
        frame_times[:movement_sz],
        events=events_matrix,
        drift_model=drift_model,
        drift_order=drift_order,
        add_regs=movement_regressor,
        add_reg_names=add_reg_names,
        hrf_model=hrf_model,
    )
    return design_mat

def convert_auto_labels(labels, fmri_time_points):
    # convert the automatic labels to the format used in the self
    proc_labels = np.zeros(fmri_time_points)
    for i in range(labels.shape[0]):
        st_ind = int(np.round(labels['onset'].values[i]))
        end_ind = int(np.round(labels['onset'].values[i] + labels['duration'].values[i]))
        proc_labels[st_ind:end_ind] = 1
    # proc_labels = proc_labels.astype(np.int16)
    return proc_labels

def build_contrast_noface_events(annots, n_scans, trial_type):
    face_labels_in_time = convert_auto_labels(annots, n_scans)
    # build events no face annotaitons
    no_face_annot = pd.DataFrame(columns=['onset', 'duration'])
    duration = 0
    onset_start = 0
    onsets = []
    durations = []
    for ind, row in enumerate(face_labels_in_time):
        if row == 0:
            # count the duration of the no face event
            onset_start = ind if duration == 0 else onset_start
            duration += 1
        else:
            if duration > 0:
                onsets.append(onset_start)
                durations.append(duration)
                duration = 0
    if duration > 0:
        onsets.append(onset_start)
        durations.append(duration)
    # padd the last event to fit the number of scans
    last_event = onsets[-1] + durations[-1]
    if last_event < n_scans:
        onsets.append(last_event)
        durations.append(n_scans - last_event)

    no_face_annot['onset'] = onsets
    no_face_annot['duration'] = durations
    no_face_annot['trial_type'] = trial_type
    return no_face_annot



if __name__ == '__main__':
    annotations_path = r"E:\NND\ds002837-download\stimuli\task-500daysofsummer_face-annotation.1D"
    movement_regressor_path = r"C:\Users\alonv\Downloads\sub-1_task-500daysofsummer_polort_bandpass_vent_wm_motion.1D"
    fmri_path = r"E:\NND\ds002837-download\derivatives\sub-1\func\sub-1_task-500daysofsummer_bold_blur_censor.nii.gz"

    use_only_6_motion_regr = True
    combine_runs = False

    fmri_data = nibabel.load(fmri_path)

    feature_1 = 'face'
    feature_2 = 'noface'

    tr = 1.0  # repetition time is 1 second
    slice_time_ref = 0.5  # we will align slice acquisition time to the middle of the TR
    n_scans = fmri_data.shape[-1]  # the acquisition comprises 128 scans
    frame_times = np.arange(n_scans) * tr  # here are the corresponding frame times
    # The 6 parameters correspond to three translations and three
    # rotations describing rigid body motion
    add_reg_names = [[f"tx_r{ind}", f"ty_r{ind}", f"tz_r{ind}", f"rx_r{ind}", f"ry_r{ind}", f"rz_r{ind}"] for ind in range(1)]
    add_reg_names = np.array(add_reg_names).flatten().tolist()
    # load annotations, fmri data and regressors for the first subject
    face_annots = pd.read_csv(annotations_path, delimiter=' ', names=['onset', 'duration'])
    face_annots['trial_type'] = feature_1
    movement_regressor = np.loadtxt(movement_regressor_path)
    if use_only_6_motion_regr:
        # use only the first 6 motion regressors, there are 18 in total (6 for each run, 3 runs in total)
        head_motion_indices = np.arange(-36, -18)
        movement_regressor = movement_regressor[:, head_motion_indices]
    if combine_runs:
        # combine the runs
        movement_regressor[:, :6] = movement_regressor[:, :6] + movement_regressor[:,6:12] + movement_regressor[:, 12:]
        movement_regressor = movement_regressor[:, :6]
    # build the no face events
    no_face_annots = build_contrast_noface_events(face_annots, n_scans, trial_type=feature_2)
    # create the design matrix
    events = pd.concat([face_annots, no_face_annots], ignore_index=True)
    events = events.sort_values(by='onset')
    events = events.reset_index(drop=True)

    # build the contrasts based on the design matrix
    # contrasts = []
    # face_vector = convert_auto_labels(events.loc[events['trial_type'] == feature_1], n_scans)
    # no_face_vector = convert_auto_labels(events.loc[events['trial_type'] == feature_2], n_scans)

    design_matrix = build_design_matrix(fmri_data, tr=tr, movement_regressor=movement_regressor, events_matrix=events)
    contrasts = np.eye(design_matrix.shape[1])
    contrasts = contrasts[0] - contrasts[1]

    plot_design_matrix(design_matrix)
    plt.show()

    voxel_slice = fmri_data.slicer[..., :design_matrix.shape[0]]
    first_level_model = FirstLevelModel(t_r=tr, slice_time_ref=slice_time_ref, mask_img=False, verbose=2,
                                        minimize_memory=False)
    first_level_model = first_level_model.fit(
        run_imgs=voxel_slice, design_matrices=design_matrix
    )

    # display the estimated signal plot vs the original signal
    # time_start = 1000
    # time_limit = 100
    # predicted = first_level_model.predicted[0].get_fdata().reshape(-1, voxel_slice.shape[-1])[0][time_start:time_start+time_limit]
    # original = voxel_slice.get_fdata().reshape(-1, voxel_slice.shape[-1])[0][time_start:time_start+time_limit]
    # plt.plot(predicted, label="predicted")
    # plt.plot(original, label="original")
    # plt.legend()
    # plt.show()


    z_map = first_level_model.compute_contrast(
        contrasts, output_type="z_score"
    )

    from nilearn.reporting import make_glm_report

    # report = make_glm_report(
    #     first_level_model,
    #     contrasts={'face_noface':contrasts},
    #     title="ADHD DMN Report",
    #     cluster_threshold=15,
    #     min_distance=8.0,
    #     plot_type="glass",
    # )
    mean_image = mean_img(voxel_slice)
    plotting.plot_stat_map(
        z_map,
        bg_img=mean_image,
        threshold=2.0,
        display_mode="z",
        cut_coords=2,
        black_bg=True,
        title='0',
    )
    plotting.show()
    pass