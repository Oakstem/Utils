import os.path
from glob import glob
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cosine

embd1_path = '/Users/alonz/PycharmProjects/merlot_reserve/demo/embeddings/single_modality_vis/combined/'
embd2_path = '/Users/alonz/PycharmProjects/merlot_reserve/demo/embeddings/single_modality_vis_empty_text/combined/'

files = glob(f'{embd1_path}*')

for file in files:
    filename = Path(file).name
    embd2_filepath = Path(embd2_path) / filename
    if os.path.exists(embd2_filepath):
        embd1 = np.load(file, allow_pickle=True)
        embd2 = np.load(embd2_filepath, allow_pickle=True)
        diff = embd1 - embd2
        diff_sum = abs(diff).sum()
        cosine_sim = cosine(embd1, embd2)
        print(f'File:{filename}, Diff:{diff_sum}, cosine_sim:{cosine_sim}')