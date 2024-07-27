import os
import shutil
from glob import glob
import numpy as np
import pandas as pd
from pathlib import Path

# lets read the annotation file, extract the filenames of relevant files, and move all
# of these files from the original directory to a new directory

annot_path = '/Users/alonz/PycharmProjects/merlot_reserve/demo/combined_annotations.csv'
data_dir = '/Volumes/swagbox/moments/Moments_in_Time_Raw'
new_dir = '/Volumes/swagbox/moments/Moments_in_Time_layla'
annot_df = pd.read_csv(annot_path)

for file in annot_df.path.values:
    if not os.path.exists(file):
        continue
    filename = Path(file).name
    new_file_path = file.replace('Moments_in_Time_Raw', 'Moments_in_Time_layla')
    Path(new_file_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(file, new_file_path)
    print(f'Moved {filename} to {new_dir}')

