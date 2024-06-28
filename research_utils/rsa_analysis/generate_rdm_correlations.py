import os
import re
from pathlib import Path
import pandas as pd
import numpy as np
import research_utils
from research_utils.rsa_analysis.rdm_generator import create_and_save_rdm
from research_utils.rsa_analysis.rsa_analysis import create_rsa_from_two_rdm_mats, fix_uneven_dataframes, \
    find_all_inner_rdm_files
from research_utils.rsa_analysis.label_correlation_utils import l1_diff, load_relevant_sts_rdms, check_rsa_against_fmri, \
    strip_file_extension, extract_subject_string, extract_roi_string
import plotly.express as px
import plotly.io as pio
import plotly.offline as pyo


class LabelCorrelation:
    """ Class for statistical comparison of model RDM results vs labels (features such as valence, arousal, etc.)"""
    def __init__(self, config):
        self.config=config
        self._init_paths(**config)
        self._load_data(**config)
        self._load_model_rdms(**config)

    def _init_paths(self, combined_annot_df_path=None, fmris_dir=None, results_path=None, **kwargs):
        if combined_annot_df_path is None:
            combined_annot_df_path = '/Users/alonz/PycharmProjects/merlot_reserve/demo/combined_annotations.csv'
        self.combined_annot_df_path = combined_annot_df_path

        if fmris_dir is None:
            fmris_dir = '/Users/alonz/PycharmProjects/pSTS_DB/psts_db/fmri'
        self.fmris_dir = fmris_dir

        if results_path is None:
            results_path = Path(research_utils.__path__[0]).parent / 'results'
        if not os.path.exists(results_path):
            Path(results_path).mkdir(parents=True, exist_ok=True)
        self.results_path = Path(results_path)

        # create a timestamp that will be attached to all results
        self.timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

    def _load_data(self,**kwargs):
        self.combined_annot_df = pd.read_csv(self.combined_annot_df_path)
        self.combined_annot_df.index = self.combined_annot_df['filename'].apply(lambda x: x.split('.')[0])
        self.labels = self.combined_annot_df.loc[:, 'indoor':'arousal'].columns.values
        # labels = [labels[0]]
        print('Loading fmri data...')
        self.fmri_rdms = load_relevant_sts_rdms(self.fmris_dir)
        self.fmri_rdms = {key: val['train'] for key, val in self.fmri_rdms.items()}

    def _load_model_rdms(self, model_db_path=None, model_keys_to_rename={}, **kwargs):
        """ search for all model RDM files in the given model_db_path directory, an RDM file is considered a file with default
        prefix of: 'all_inner_'"""
        if model_db_path is None:
            model_db_path = '/Users/alonz/PycharmProjects/pSTS_DB/psts_db'

        paths_dd = find_all_inner_rdm_files(model_db_path)
        self.model_rdms = {key: pd.read_csv(val, index_col=0) for key, val in paths_dd.items()}
        # strip file extension from index / columns
        self.model_rdms = strip_file_extension(self.model_rdms)

        for key, new_key in model_keys_to_rename.items():
            self.model_rdms = self.rename_model_key(self.model_rdms, key, new_key)

    def filter_relevant_models(self, relevant_models, **kwargs):
        self.model_rdms = {key: val for key, val in self.model_rdms.items() if key in relevant_models}

    def rsa_models_against_labels(self, **kwargs):
        if getattr(self, 'labels', None) is None:
            raise ValueError('Labels must be defined before running rsa_models_against_labels')

        full_results = {}
        rdm_results_dir = self.results_path / 'RDM'
        rdm_results_dir.mkdir(parents=True, exist_ok=True)

        # build feature distances RDM matrix
        for feature in self.labels:
            rdm_result_path =rdm_results_dir /  f'{feature}_rdm.csv'
            feature_df = self.combined_annot_df[feature]
            feature_embd = feature_df.to_dict()
            feature_rdm = create_and_save_rdm(feature_embd, feature_df.index, l1_diff, rdm_result_path, similarity=True)
            rsa_results = check_rsa_against_fmri(self.model_rdms, feature_rdm, [feature])
            full_results.update(rsa_results)
        full_results = pd.DataFrame([full_results]).T.sort_values(by=0)
        return full_results


    def split_results_by_subject_and_feature(self, full_results):
        full_results['feature'] = [val.split('/')[-1] for val in full_results.index.values]
        full_results['subject'] = [f"{extract_subject_string(val)}/{extract_roi_string(val)}" for val in
                                   full_results.index.values]
        return full_results

    def fix_model_namings(self, full_results, strip_vals=None):
        if strip_vals is None:
            strip_vals = ['_vitB_32', '_RDM', '_attentinve_pooler_out']
        for strip_val in strip_vals:
            full_results['model'] = full_results['model'].str.replace(strip_val, '')
        return full_results

    def plot_features_vs_models(self, full_results):
        save_path = self.results_path / 'plots' / f'{self.timestamp}_models_vs_features.html'
        save_path.parent.mkdir(parents=True, exist_ok=True)

        full_results['model'] = [val.split('/')[0] for val in full_results.index.values]
        # strip unnessecary parts of the model name
        full_results = self.fix_model_namings(full_results)
        full_results = full_results.sort_values(by='model')

        fig = px.bar(full_results, x='model', y=0, color='feature', barmode='group')

        fig.update_yaxes(title_text='correlation')
        fig.show()
        pio.write_html(fig, save_path)

    @staticmethod
    def rename_model_key(model_rdms, key, new_key):
        model_rdms[new_key] = model_rdms.pop(key)
        return model_rdms

if __name__ == "__main__":
    # create model rdm's vs label(features) correlation plots
    # load data
    config = {'combined_annot_df_path': '/Users/alonz/PycharmProjects/merlot_reserve/demo/combined_annotations.csv',
              'model_db_path': '/Users/alonz/PycharmProjects/pSTS_DB/psts_db',
              'model_keys_to_rename':
                  {'clip_expert-pond-125_20240618_220556_expert-pond-125_pyhlu5d1_image': 'clip_finetuned_visual'},
              'relevant_models':['mreserve_unimodal_audio_combined',
                                'mreserve_unimodal_visual_combined',
                                'mreserve_multimodal_combined',
                                'resnet50_normal_visual',
                                'clip_vitB_32_visual',
                                'vjepa_attentinve_pooler_out_visual',
                                'imagebind_RDM_vision',
                                'imagebind_RDM_audio',
                                'imagebind_RDM_text',
                                'clip_finetuned_visual']}

    label_correlator = LabelCorrelation(config)
    label_correlator.filter_relevant_models(**config)
    rsa_results = label_correlator.rsa_models_against_labels(**config)
    rsa_results = label_correlator.split_results_by_subject_and_feature(rsa_results)
    label_correlator.plot_features_vs_models(rsa_results)
    pass