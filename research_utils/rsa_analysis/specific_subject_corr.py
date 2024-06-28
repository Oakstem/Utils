import os
import re
from pathlib import Path

import pandas as pd
import numpy as np
from research_utils.rsa_analysis.rdm_generator import create_and_save_rdm
from research_utils.rsa_analysis.rsa_analysis import create_rsa_from_two_rdm_mats, fix_uneven_dataframes, \
    find_all_inner_rdm_files
import plotly.express as px
import plotly.io as pio
import plotly.offline as pyo

#%%
#L1 diff
def l1_diff(vec1, vec2):
    return np.sum(np.abs(vec1-vec2))

def load_relevant_sts_rdms(path):
    rdms_dd = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if 'psts' in file.lower():
                csv_path = Path(os.path.join(root, file))
                stage = csv_path.parts[-3]
                embd = pd.read_csv(csv_path, index_col=0)
                if file in rdms_dd:
                    rdms_dd[file].update({stage: embd})
                else:
                    rdms_dd[file] = {stage: embd}
                if 'sub-03_task-FBOS_space' in file:
                    print(f'###### Loaded {file} ######')
    return rdms_dd

def check_rsa_against_fmri(fmri_rdms, feature_rdm, labels):
    rsa_results = {}
    for fmri_name, fmri_rdm in fmri_rdms.items():
        for label in labels:
            fmri_rdm, feature_rdm = fix_uneven_dataframes(fmri_rdm, feature_rdm)
            rsa_corr = create_rsa_from_two_rdm_mats(fmri_rdm, feature_rdm, fmri_name, label,
                                                    'rsa', 'full_corr_rsa.csv',
                                                    'partial_corr_rsa.csv')
            rsa_results[f"{fmri_name}/{label}"] = rsa_corr[0]['r'].values[0]
    return rsa_results

def plot_results(full_results, title, filename, color='feature', y=0, x='index'):
    fig = px.bar(full_results, x=x, y=y, color=color, labels={'feature':'Feature'}, title=title)
    fig.update_yaxes(title_text='correlation')
    pyo.iplot(fig)
    fig.show()
    pio.write_html(fig, filename)

def extract_task_string(s):
    match = re.search(r'task-(.*?)_space', s)
    return match.group(1) if match else None

def extract_roi_string(s):
    match = re.search(r'roi-(.*?)_hemi', s)
    return match.group(1) if match else None

def extract_subject_string(s):
    match = re.search(r'(sub-\d\d)', s)
    return match.group(1) if match else None

def strip_file_extension(rdm_dict):
    # strip file extension from index / columns
    for key, val in rdm_dict.items():
        val.index = [x.split('.')[0] for x in val.index]
        val.columns = [x.split('.')[0] for x in val.columns]
    return rdm_dict

#%%