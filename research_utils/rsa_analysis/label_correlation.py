import os
import re
from pathlib import Path

import pandas as pd
import numpy as np
from research_utils.rsa_analysis.rdm_generator import create_and_save_rdm
from research_utils.rsa_analysis.rsa_analysis import create_rsa_from_two_rdm_mats, fix_uneven_dataframes, \
    find_all_inner_rdm_files
from research_utils.rsa_analysis.label_correlation_utils import l1_diff, load_relevant_sts_rdms, check_rsa_against_fmri
import plotly.express as px
import plotly.io as pio
import plotly.offline as pyo


#%% Load data
combined_annot_df_path = '/Users/alonz/PycharmProjects/merlot_reserve/demo/combined_annotations.csv'
fmris_dir = '/Users/alonz/PycharmProjects/pSTS_DB/psts_db/fmri'
results_path = Path('./results')
results_path.mkdir(parents=True, exist_ok=True)
combined_annot_df = pd.read_csv(combined_annot_df_path)
combined_annot_df.index = combined_annot_df['filename'].apply(lambda x: x.split('.')[0])
labels = combined_annot_df.loc[:, 'indoor':'arousal'].columns.values
# labels = [labels[0]]
print('Loading fmri data...')
fmri_rdms = load_relevant_sts_rdms(fmris_dir)
fmri_rdms = {key: val['train'] for key, val in fmri_rdms.items()}
#%% Split fmri by subjects
# sub = 'sub-04'
# fmri_rdms = {key: val for key, val in fmri_rdms.items() if sub in key}

#%% Load model RDM data
model_db_path = '/Users/alonz/PycharmProjects/pSTS_DB/psts_db'
paths_dd = find_all_inner_rdm_files(model_db_path)
model_rdms = {key: pd.read_csv(val, index_col=0) for key, val in paths_dd.items()}
# strip file extension from index / columns
model_rdms = strip_file_extension(model_rdms)

def rename_model_key(key, new_key):
    model_rdms[new_key] = model_rdms.pop(key)
    return model_rdms

model_rdms = rename_model_key('clip_expert-pond-125_20240618_220556_expert-pond-125_pyhlu5d1_image', 'clip_finetuned_visual')

#%% Filter relevant models
relevant_models = [
 'mreserve_unimodal_audio_combined',
 'mreserve_unimodal_visual_combined',
 'mreserve_multimodal_combined',
 'resnet50_normal_visual',
 'clip_vitB_32_visual',
 'vjepa_attentinve_pooler_out_visual',
 'imagebind_RDM_vision',
 'imagebind_RDM_audio',
 'imagebind_RDM_text',
'clip_finetuned_visual']
model_rdms = {key: val for key, val in model_rdms.items() if key in relevant_models}

#%% Run RSA between features & models / fmri
full_results = {}

# build feature distances RDM matrix
for feature in labels:
    rdm_result_path = results_path / f'{feature}_rdm.csv'
    feature_df = combined_annot_df[feature]
    feature_embd = feature_df.to_dict()
    feature_rdm = create_and_save_rdm(feature_embd, feature_df.index, l1_diff, rdm_result_path, similarity=True)
    rsa_results = check_rsa_against_fmri(model_rdms, feature_rdm, [feature])
    full_results.update(rsa_results)
full_results = pd.DataFrame([full_results]).T.sort_values(by=0)

#%% Run RSA between fmri & models
model_rdms = strip_file_extension(model_rdms)
fmri_rdms = strip_file_extension(fmri_rdms)
full_results = {}
# full_results = check_rsa_fmri_against_models(fmri_rdms, model_rdms)
rsa_results = {}
for fmri_name, fmri_rdm in fmri_rdms.items():
    for strip_val, model_rdm in model_rdms.items():
        fmri_rdm, model_rdm = fix_uneven_dataframes(fmri_rdm, model_rdm)
        rsa_corr = create_rsa_from_two_rdm_mats(fmri_rdm, model_rdm, fmri_name, strip_val,
                                                'rsa', 'full_corr_rsa.csv',
                                                'partial_corr_rsa.csv')
        rsa_results[f"{fmri_name}/{strip_val}"] = rsa_corr[0]['r'].values[0]
# full_results = pd.DataFrame([rsa_results]).T.sort_values(by=0)
#%%
full_results['feature'] = [val.split('/')[-1] for val in full_results.index.values]
full_results['subject'] = [f"{extract_subject_string(val)}/{extract_roi_string(val)}" for val in full_results.index.values]
#%% Plot results
# Plot FMRI vs Features for all features & subjects / regions
plot_results(full_results, 'RSA results for FMRI vs Features', 'fmri_vs_features.html')
#%%
# Group by feature
grouped_by_feature = full_results.groupby('feature').mean().sort_values(by=0) # mean of all fmri results for each feature
plot_results(grouped_by_feature, 'Grouped by feature', 'fmriGrouped_vs_features_mean.html')
#%%
# Group by fmri region
full_results['feature'] = [val.split('/')[0] for val in full_results.index.values]
grouped_by_fmri = full_results.groupby('feature').mean().sort_values(by=0) # mean of all feature results for each fmri
plot_results(grouped_by_fmri, 'Grouped by fmri', 'fmriGrouped_vs_features_mean.html')
#%%
# Group by subject
# grouped_by_feature = full_results.groupby('feature').mean().sort_values(by=0)
# grouped_by_subject = grouped_by_feature.groupby(full_results['subject']).mean().sort_values()
plot_results(full_results, 'Grouped by subject', 'fmriGrouped_vs_features_mean.html',
             color='feature', y=0, x='subject')
# pio.write_html(fig, filename)

#%% Model Evaluation
# Plot Model vs Features for all features & subjects / regions
# sort by model name
# take only the first name of the model name string
full_results['model'] = [val.split('/')[0] for val in full_results.index.values]
# strip unnessecary parts of the model name
strip_vals = ['_vitB_32', '_RDM', '_attentinve_pooler_out']
for strip_val  in strip_vals:
    full_results['model'] = full_results['model'].str.replace(strip_val, '')

full_results = full_results.sort_values(by='model')
# reset index

# full_results = full_results.reset_index()
fig = px.bar(full_results, x='model', y=0, color='feature', barmode='group')
# fig = px.bar(full_results, x='feature', y=0, color='model', barmode='group')
fig.update_yaxes(title_text='correlation')
pio.write_html(fig, 'models_vs_features.html')
# plot_results(full_results, 'RSA results for Model vs Features', 'models_vs_features.html', x=full_results.index)

#%% Model vs FMRI Evaluation
full_results = pd.DataFrame([rsa_results]).T.sort_values(by=0)
full_results['roi'] = [extract_roi_string(val) for val in full_results.index.values]
full_results['model_type'] = [val.split('/')[1] for val in full_results.index.values]
# grouped_by_fmri = full_results.groupby('roi').mean().sort_values(by=0)  # mean of all feature results for each fmri
# grouped_by_fmri['subject'] = [f"{extract_subject_string(val)}/{extract_roi_string(val)}" for val in grouped_by_fmri.index.values]
plot_results(full_results, 'RSA results for FMRI vs Models', results_path / f'models_vs_fmri_{sub}.html', x=full_results.roi, color=full_results.model_type)
