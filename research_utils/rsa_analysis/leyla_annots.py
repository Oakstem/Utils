import os
from pathlib import Path

import pandas as pd

def append_filename(df):
    filenames = df.iloc[:, 0].str.split('/', expand=True)
    df['filename'] = filenames.iloc[:,1].values
    return df
def find_video_in_moments(filename):
    ind = mm_df['filename'].isin([filename])
    mm_row = mm_df.loc[ind]
    return mm_row

# Lets load an annotation file with all the type of activities possible for the social videos
leyla_annot_path = "/Users/alonz/PycharmProjects/SIfMRI_analysis/annotations.csv"
moments_train_path = "/Volumes/swagbox/moments/Moments_in_Time_Raw/trainingSet.csv"
moments_val_path = "/Volumes/swagbox/moments/Moments_in_Time_Raw/validationSet.csv"
data_origin = Path(moments_train_path).parent.__str__()
combined_df_path = "combined_annotations.csv"


annot_df = pd.read_csv(leyla_annot_path)
mm_train = pd.read_csv(moments_train_path)
mm_val = pd.read_csv(moments_val_path)

mm_cols = ['path', 'label','nb_agree', 'nb_disagree']
mm_train.columns = mm_cols
mm_val.columns = mm_cols

mm_train = append_filename(mm_train)
mm_val = append_filename(mm_val)

mm_train['path'] = mm_train['path'].apply(lambda x: os.path.join(data_origin, 'training', x))
mm_val['path'] = mm_val['path'].apply(lambda x: os.path.join(data_origin, 'validation', x))

mm_df = pd.concat([mm_train, mm_val], ignore_index=True, axis=0)


combined = pd.DataFrame([], columns=annot_df.columns.tolist() + mm_df.columns.tolist())
not_found = pd.DataFrame([], columns=annot_df.columns.tolist())

for ind, row in annot_df.iterrows():
    # lets find if the file exists in the dir
    mm_row = find_video_in_moments(row.video_name)
    if mm_row.shape[0] > 0:
        combined_row = pd.concat([row, mm_row.iloc[0,:]], axis=0, ignore_index=True)
        combined.loc[len(combined)] = combined_row.values
    else:
        not_found.loc[len(not_found)] = row

combined.to_csv(combined_df_path, index=False)
print(f"Full annotated csv saved at:{combined_df_path}")
pass