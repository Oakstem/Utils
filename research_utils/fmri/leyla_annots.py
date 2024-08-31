import glob
import os

import nibabel
import pandas as pd
import numpy as np
from nilearn import image
from scipy import stats
from nilearn.image import new_img_like, resample_to_img

import scipy.io

def read_mat_file(file_path):
    """
    Reads a .mat file and returns the data.

    Parameters:
    file_path (str): The path to the .mat file.

    Returns:
    dict: A dictionary containing the data from the .mat file.
    """
    return scipy.io.loadmat(file_path)

def convert_auto_labels(labels, func):
    # convert the automatic labels to the format used in the self
    proc_labels = np.zeros(func.shape[-1])
    for i in range(labels.shape[0]):
        st_ind = int(np.round(labels['onset'][i]))
        end_ind = int(np.round(labels['onset'][i] + labels['duration'][i]))
        proc_labels[st_ind:end_ind] = 1 if labels.shape[1] == 2 else labels.iloc[i, 2]
    return proc_labels

def convert_binary_annotations_to_events(annotations, frames_from_start=13, frames_from_end=88):
    # each annotation represents 3 seconds
    events = []
    seconds = 0
   # add first 39 seconds of no annotations, 13 frames
    for i in range(frames_from_start):
        events.append([i, 1, 0])
        seconds += 1
    for i in range(len(annotations)):
        # if annotations[i] >=0.5:
        events.append([seconds, 3, annotations[i]])
        # else:
        #     events.append([seconds, 3, annotations[]])
        seconds += 3
    # add last 264 seconds of no annotations, 88 frames
    for i in range(frames_from_end):
        events.append([seconds, 3, 0])
        seconds += 3
    events = pd.DataFrame(events, columns=['onset', 'duration', 'trial_type'])
    return events

if __name__ == '__main__':
    original_annots = '/Users/alonz/Downloads/face_events.csv'
    layla_annots = '/Users/alonz/Downloads/leyla_annotations/'
    original_mat_file = '/Users/alonz/Downloads/leyla_annotations/face.mat'
    nnd_labels_path = '/Users/alonz/PycharmProjects/pSTS_DB/psts_db/datasets/NND/stimuli/task-500daysofsummer_face-annotation.1D'
    func_file = nibabel.load('/Users/alonz/PycharmProjects/pSTS_DB/psts_db/datasets/NND/derivatives/sub-1/func/sub-1_task-500daysofsummer_bold_blur_censor_ica.nii.gz')
    results_dir = '/Users/alonz/Downloads/leyla_annotations/adjusted/'
    os.makedirs(results_dir, exist_ok=True)
    # read mat file annotations
    nnd_annots =  pd.read_csv(nnd_labels_path, delimiter=' ', names=['onset', 'duration'])
    nnd_annots = convert_auto_labels(nnd_annots, func_file)
    # processed_layla_annots = pd.read_csv(original_annots)
    original_layla_file = read_mat_file(original_mat_file)

    def get_data(mat_file):
        res = [[key, val] for key, val in mat_file.items() if isinstance(val, np.ndarray)]
        if len(res) != 1:
            raise ValueError(f"Expected one array in the mat file, got {len(res)}")
        res = [res[0][0], res[0][1].reshape(-1)]
        return res

    ## Process all annotations in the dir
    for file in glob.glob(f"{layla_annots}/*.mat"):
        original_layla_file = read_mat_file(file)
        key, original_layla_annot = get_data(original_layla_file)
        processed_layla_annot = convert_binary_annotations_to_events(annotations=original_layla_annot, frames_from_start=39)
        processed_layla_annot = pd.DataFrame(processed_layla_annot.values, columns=['onset', 'duration', 'type'])
        # processed_layla_annot = processed_layla_annot.loc[processed_layla_annot['type'] == 1]
        processed_layla_annot.index = range(processed_layla_annot.shape[0])
        processed_layla_annot = pd.DataFrame(convert_auto_labels(processed_layla_annot, func_file))
        processed_layla_annot.to_csv(f"{results_dir}/{key}.csv", index=False)


    # Lets adapt the layla annotations to the NND annotations format - same dimension of the func file with ones and zeros
    processed_layla_annots = convert_binary_annotations_to_events(annotations=original_layla_file['face'], frames_from_start=39)
    processed_layla_annots = pd.DataFrame(processed_layla_annots.values, columns=['onset', 'duration', 'type'])
    processed_layla_annots = processed_layla_annots.loc[processed_layla_annots['type'] == 1]
    processed_layla_annots.index = range(processed_layla_annots.shape[0])
    processed_layla_annots = convert_auto_labels(processed_layla_annots, func_file)

    # find correlation between the two annotations
    corr = np.corrcoef(nnd_annots, processed_layla_annots)

    # Lets plot with plotly the two annotations
    import plotly.express as px
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=nnd_annots, mode='lines', name='NND'))
    fig.add_trace(go.Scatter(y=processed_layla_annots, mode='lines', name='Layla'))
    fig.show()


    pass
