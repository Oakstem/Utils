import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pingouin as pg
import psts_db


def create_rsa_from_two_rdms(path_to_first_rdm, path_to_second_rdm
                             ,first_rdm_type, second_rdm_type, rsa_type,
                             file_to_save_full_corr_rsa, file_to_save_partial_corr_rsa):
    rdm1 = pd.read_csv(path_to_first_rdm, index_col=0)
    rdm2 = pd.read_csv(path_to_second_rdm, index_col=0)
    rdm1, rdm2 = fix_uneven_dataframes(rdm1, rdm2)
    rdm1, rdm2 = rdm1.values, rdm2.values
    n = rdm1.shape[0]
    indices = np.triu_indices(n, k=1)  # Exclude main diagonal
    lower_triangle_first_rdm = rdm1[indices]
    lower_triangle_second_rdm = rdm2[indices]

    full_df = pd.DataFrame({first_rdm_type: lower_triangle_first_rdm, second_rdm_type: 1-lower_triangle_second_rdm})

    full_corr_results = pg.corr(lower_triangle_first_rdm, lower_triangle_second_rdm, method='pearson')
    full_corr_results.insert(0, 'Label', rsa_type)
    print(full_corr_results)
    partial_corr_result = pg.partial_corr(data=full_df, x=first_rdm_type, y=second_rdm_type)
    partial_corr_result.insert(0, 'Label', rsa_type)
    print(partial_corr_result)

    try:
        # Read the existing Excel file into a DataFrame
        full_corr_data = pd.read_csv(file_to_save_full_corr_rsa)
        partial_corr_data = pd.read_csv(file_to_save_partial_corr_rsa)

    except FileNotFoundError:
        # Create a new DataFrame if the file doesn't exist
        full_corr_data = pd.DataFrame()
        partial_corr_data = pd.DataFrame()

    # Concatenate the existing data and new results
    combined_full_corr_data = pd.concat([full_corr_data, full_corr_results], ignore_index=True)
    combined_partial_corr_data = pd.concat([partial_corr_data, partial_corr_result], ignore_index=True)

    # Save the combined data to a new sheet in the Excel file
    # combined_full_corr_data.to_csv(file_to_save_full_corr_rsa, index=False)
    # combined_partial_corr_data.to_csv(file_to_save_partial_corr_rsa, index=False)

    return combined_full_corr_data, combined_partial_corr_data

def create_rsa_from_two_rdm_mats(rdm1, rdm2
                             ,first_rdm_type, second_rdm_type, rsa_type,
                             file_to_save_full_corr_rsa, file_to_save_partial_corr_rsa):
    rdm1, rdm2 = fix_uneven_dataframes(rdm1, rdm2)
    rdm1, rdm2 = rdm1.values, rdm2.values
    n = rdm1.shape[0]
    indices = np.triu_indices(n, k=1)  # Exclude main diagonal
    lower_triangle_first_rdm = rdm1[indices]
    lower_triangle_second_rdm = rdm2[indices]
    print(f'rdm1 shape: {rdm1.shape}, rdm2 shape: {rdm2.shape}')
    full_df = pd.DataFrame({first_rdm_type: lower_triangle_first_rdm, second_rdm_type: 1-lower_triangle_second_rdm})

    full_corr_results = pg.corr(lower_triangle_first_rdm, lower_triangle_second_rdm, method='pearson')
    full_corr_results.insert(0, 'Label', rsa_type)
    print(full_corr_results)
    partial_corr_result = pg.partial_corr(data=full_df, x=first_rdm_type, y=second_rdm_type)
    partial_corr_result.insert(0, 'Label', rsa_type)
    print(partial_corr_result)

    try:
        # Read the existing Excel file into a DataFrame
        full_corr_data = pd.read_csv(file_to_save_full_corr_rsa)
        partial_corr_data = pd.read_csv(file_to_save_partial_corr_rsa)

    except FileNotFoundError:
        # Create a new DataFrame if the file doesn't exist
        full_corr_data = pd.DataFrame()
        partial_corr_data = pd.DataFrame()

    # Concatenate the existing data and new results
    combined_full_corr_data = pd.concat([full_corr_data, full_corr_results], ignore_index=True)
    combined_partial_corr_data = pd.concat([partial_corr_data, partial_corr_result], ignore_index=True)

    # Save the combined data to a new sheet in the Excel file
    # combined_full_corr_data.to_csv(file_to_save_full_corr_rsa, index=False)
    # combined_partial_corr_data.to_csv(file_to_save_partial_corr_rsa, index=False)

    return combined_full_corr_data, combined_partial_corr_data


def fix_uneven_dataframes(df1, df2):
    if df1.shape != df2.shape:
        if df1.shape[0] < df2.shape[0]:
            df1Index = [val for val in df1.index if val in df2.index]
            df2 = df2.loc[df1Index, df1Index]
            df1 = df1.loc[df1Index, df1Index]
        else:
            df2Index = [val for val in df2.index if val in df1.index]
            df1 = df1.loc[df2Index, df2Index]
            df2 = df2.loc[df2Index, df2Index]
    return df1, df2


def normalize_df(df):
    min_val = df.min().min()
    max_val = df.max().max()

    df = (df - min_val) / (max_val - min_val)

    return df

def sort_rdm(df):
    cols = df.columns.values
    df_sorted = df.sort_values(by=cols[0])
    return df_sorted

def plot_rdm(rsa_path):
    correlation_df = pd.read_csv(rsa_path, index_col=0)
    correlation_df = normalize_df(correlation_df)
    # Create a heatmap of the correlation matrix
    plt.figure(figsize=(8, 6))
    plt.title("Correlation Matrix")
    plt.imshow(correlation_df.values, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    # Set tick labels
    plt.xticks(range(len(correlation_df.columns)), correlation_df.columns, rotation='vertical')
    plt.yticks(range(len(correlation_df.index)), correlation_df.index)
    plt.tight_layout()
    plt.show()

def find_all_inner_rdm_files(path, prefix='all_inner_'):
    """
    Find all the inner RDM files in the given path
    :param path: Path to search for the files
    :param prefix: Prefix of the file to search for
    :return: Dictionary with the paths of the RDM files
    """
    paths_dd = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if prefix in file:
                p = Path(root)
                modality = p.name
                model_setup = p.parent.name
                model = p.parent.parent.name
                key_path = '_'.join([model, model_setup, modality])
                paths_dd[key_path] = p / file
    return paths_dd

if __name__ == '__main__':
    use_paths_dd = True
    db_path = Path(psts_db.__path__[0])
    project = 'mreserve'
    model_setup = 'single_modality_vis_masked_text'
    results_path = db_path / project / model_setup
    results_path = db_path


    # paths_dd = {#'comb_audio': results_path / 'audio' / 'all_inner_cosine_similarity.csv',
    #              'comb_video': results_path / 'visual' / 'all_inner_cosine_similarity.csv',
    #             'comb_joint': results_path / 'combined' / 'all_inner_cosine_similarity.csv'}
    #             'category_audio': '/Users/alonz/PycharmProjects/pSTS_DB/psts_db/mreserve/combined_mask_at_start_embeddings/audio_combined_RDM_cosine.csv',
    #             'category_video': '/Users/alonz/PycharmProjects/pSTS_DB/psts_db/mreserve/combined_mask_at_start_embeddings/visual_combined_RDM_cosine.csv',
    #             'category_combined': '/Users/alonz/PycharmProjects/pSTS_DB/psts_db/mreserve/combined_mask_at_start_embeddings/combined_combined_RDM_cosine.csv',
    #             'category_text': '/Users/alonz/PycharmProjects/pSTS_DB/psts_db/mreserve/combined_mask_at_start_embeddings/text_combined_RDM_cosine.csv'}
                # 'comb_text': '/Users/alonz/PycharmProjects/merlot_reserve/demo/RDM/combined_mask_at_start_embeddings/text_combined_RDM_cosine.csv',
                # 'single_video': '/Users/alonz/PycharmProjects/merlot_reserve/demo/RDM/single_modality_vis_masked_text/visual_combined_RDM_cosine.csv',
                # 'single_video_joint': '/Users/alonz/PycharmProjects/merlot_reserve/demo/RDM/single_modality_vis_masked_text/combined_combined_RDM_cosine.csv',
                # 'single_audio': '/Users/alonz/PycharmProjects/merlot_reserve/demo/RDM/single_modality_audio/audio_combined_RDM_cosine.csv',
                # 'single_audio_joint': '/Users/alonz/PycharmProjects/merlot_reserve/demo/RDM/single_modality_audio/combined_combined_RDM_cosine.csv',
                # 'encoder_out': '/home/gentex/PycharmProjects/torch/data/vjepa/RDM/embeddings/encoder_out/all_inner_cosine_similarity.csv',
                # 'attentive_pool_out': '/home/gentex/PycharmProjects/torch/data/vjepa/RDM/embeddings/attentinve_pooler_out/all_inner_cosine_similarity.csv'}
    paths_dd = find_all_inner_rdm_files(db_path)

    results_dir = Path(results_path) / 'RSA'
    results_dir.mkdir(parents=True, exist_ok=True)
    full_corr_res_path = results_dir / 'full_corr.csv'
    partial_corr_res_path = results_dir / 'partial_corr.csv'
    # modalities = ['comb_audio', 'comb_video', 'comb_joint']
    modalities = list(paths_dd.keys())
    # modalities = list(paths_dd.keys())

    full_corr_mat = np.zeros((len(modalities), len(modalities)))
    for ind1, mod1 in enumerate(modalities):
        for ind2, mod2 in enumerate(modalities):
            if not use_paths_dd:
                mod1_path = f'{results_path}/{mod1}_combined_RDM_cosine.csv'
                mod2_path = f'{results_path}/{mod2}_combined_RDM_cosine.csv'
            else:
                mod1_path = paths_dd[mod1]
                mod2_path = paths_dd[mod2]

            if mod1 != mod2:
                full, partial = create_rsa_from_two_rdms(mod1_path, mod2_path, mod1, mod2,
                                         'first_rsa', full_corr_res_path, partial_corr_res_path)
                full_corr_mat[ind1, ind2] = full.r.mean()
            else:
                full_corr_mat[ind1, ind2] = 1.

    res_df = pd.DataFrame(data=full_corr_mat, columns=modalities, index=modalities)
    res_df.to_csv(full_corr_res_path)
    print(f'Results saved at:{full_corr_res_path}')
    pass