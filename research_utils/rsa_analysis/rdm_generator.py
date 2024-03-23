import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pingouin as pg
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine
from tqdm import tqdm


def pearson(vec1, vec2):
    res = pearsonr(vec1, vec2)
    return res.correlation

def euclidean(vec1, vec2):
    res = np.linalg.norm(vec2-vec1, ord=2)
    return res

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

def create_and_save_rdm(embeddings, labels, measure_func, dest_path):
    rdm = np.zeros((len(embeddings), len(embeddings)))
    filenames = list(embeddings.keys())
    progr = tqdm(enumerate(embeddings.values()))
    for i, first_em in progr:
        progr.set_description(f'Calculating RDM matrix {i}/{len(embeddings.values())}')
        for j, second_em in enumerate(embeddings.values()):
            rdm[i, j] = measure_func(first_em, second_em)

    n = len(labels)
    rdm_df = pd.DataFrame(data=rdm, columns=filenames, index=filenames)
    rdm_df.to_csv(dest_path)

    print(f"rdm saved successfuly at {dest_path}")

def flatten_if_needed(np_arr: np.ndarray):
    if len(np_arr.shape) > 1:
        np_arr = np_arr.flatten()
    return np_arr

def load_embeddings_in_dir(path, relevant_filenames):
    files = glob.glob(path + '/*.npy')
    filenames = [Path(val).stem for val in files]
    relevant_filenames = [Path(val).stem for val in relevant_filenames]
    files = [name for ind, name in enumerate(files) if np.isin(filenames[ind], relevant_filenames).any()]
    embd_arr = {}
    for file in files:
        try:
            embd = np.load(file)
            filename = Path(file).name
            embd_arr[filename] = flatten_if_needed(embd)
        except:
            print(f'Couldnt load file:{file}')
    return embd_arr

def robust_scaling(data, apply):
    if apply:
        data = np.c_[data]
        one_vec_data = data.reshape(-1)
        # Calculate the first and third quartiles
        q1 = one_vec_data.min()
        q3 = one_vec_data.max()

        # Calculate the interquartile range (IQR)
        iqr = abs(q3 - q1)

        # Perform Robust Scaling
        scaled_data = (data - q1) / iqr
        print(f"Scaled data: mean:{scaled_data.mean()}, max:{scaled_data.max()}, min:{scaled_data.min()}")
        # Turn back to list of numpy arrays
        data = [np.array(val) for val in scaled_data.tolist()]


    return data

def get_labels(df, social, th=0.7):
    if social:
        communicative = df.communication >= th
        non_communicative = df.communication < 1 -th
        df.loc[communicative, 'label'] = 'communicating'
        df.loc[non_communicative, 'label'] = 'non_communicative'
        df.loc[(~communicative) & (~non_communicative) , 'label'] = 'uncertain'

    labels = df.label.unique()

    return labels, df



def get_relevant_filenames(annot_df, labels):
    filenames = annot_df.loc[annot_df.label.isin(labels), 'filename']
    return filenames

def get_embd_dirname(social):
    dirname = 'embeddings' if not social else 'social_embeddings'
    return dirname

def normalize_df(df):
    min_val = df.min().min()
    max_val = df.max().max()

    df = (df - min_val) / (max_val - min_val)

    return df

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
    plt.savefig(rsa_path.__str__().replace(rsa_path.suffix,'.png'))
    plt.show()

def build_results_path(main_dir_path, embd_dirname, label, measure_func):
    results_path = Path(main_dir_path) / 'RDM' / embd_dirname / modality / f'{label}_inner_{measure_func.__name__}.csv'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    return results_path

def action_class_loop(annot_df, embd_dir, main_dir_path, measure_func):
    embd_dirname = Path(embd_dir).parent.name
    for label in labels:
        filenames = get_relevant_filenames(annot_df, [label])
        if len(filenames) > 2:
            embd_arr = load_embeddings_in_dir(embd_dir, filenames)
            results_path = build_results_path(main_dir_path, embd_dirname, label, measure_func)
            # Build Inner class RDM
            create_and_save_rdm(embd_arr, len(embd_arr)*[label], measure_func, results_path)
            # Save an average of all representations of this class
            avg_embd[label] = np.array(embd_arr).mean(axis=0)
            # print(f'{label}: sum:{np.array(embd_arr).sum()}')

def filename_group_loop(annot_df, embd_dir, measure_func):
    label = 'all'
    unq_filenames = annot_df.filename.unique()
    embd_dirname = Path(embd_dir).parent.name
    # for filename in unq_filenames:
    embd_arr = load_embeddings_in_dir(embd_dir, unq_filenames)
    results_path = build_results_path(main_dir_path, embd_dirname, label, measure_func)
    create_and_save_rdm(embd_arr, unq_filenames, measure_func, results_path)
    pass


if __name__ == '__main__':
    apply_scaling = False
    social = False
    action_class_group = False
    # label_type = 'social' if social else 'normal'
    modality = 'encoder_out'
    # embedding_dir = get_embd_dirname(social)
    embedding_dir = 'combined_mask_at_start_embeddings'
    main_dir_path = '/home/gentex/PycharmProjects/torch/data/vjepa'
    # embd_dir = f'/Users/alonz/PycharmProjects/merlot_reserve/demo/embeddings/{embedding_dir}/' + modality
    embd_dir = f'/home/gentex/PycharmProjects/torch/data/vjepa/embedding_analysis/embeddings/' + modality
    annot_path = '/home/gentex/PycharmProjects/torch/data/combined_annotations.csv'

    annot_df = pd.read_csv(annot_path)

    labels, annot_df = get_labels(annot_df, social)
    avg_embd = {}
    # labels = annot_df.label.unique()
    measure_func = cosine_similarity

    if not action_class_group:
        filename_group_loop(annot_df, embd_dir, measure_func)
    else:
        action_class_loop(annot_df, embd_dir, main_dir_path, measure_func)
        # Now lets build and RDM between different labels
        results_path = Path(main_dir_path) / 'RDM' / embedding_dir /f'{modality}_combined_RDM_{measure_func.__name__}.csv'
        embd_arr = list(avg_embd.values())
        label_keys = list(avg_embd.keys())
        embd_arr = robust_scaling(embd_arr, apply_scaling)
        create_and_save_rdm(embd_arr, label_keys, measure_func, results_path)

        # label_embds = np.load('/Users/alonz/PycharmProjects/merlot_reserve/demo/0_5seg_embeddings/text/labels.npy')


        plot_rdm(results_path)
