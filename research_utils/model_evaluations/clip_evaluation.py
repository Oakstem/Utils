from pathlib import Path

from clip_finetune.eval import load_model_from_checkpoint
from torchvision.io import read_image, ImageReadMode

from research_utils.rsa_analysis.embed_save import EmbeddingSave
import pandas as pd
import yaml
import numpy as np
import research_utils
import os
from clip_finetune.eval_tools import load_config_and_prepare_for_eval
import torch
import glob


def get_img_paths(df, dir_path=None):
    if dir_path is None:
        dir_path = "/Users/alonz/PycharmProjects/data/moments/moments_images/Moments_in_Time_Raw"
    new_paths = []
    for ind, row in df.iterrows():
        fname = Path(row.video_name).stem
        img_paths = glob.glob(f"{dir_path}/*/{fname}.png")
        if len(img_paths) > 0:
            new_paths.append(img_paths[0])
        else:
            new_paths.append(None)
    df['path'] = new_paths
    return df

def load_image(image_path, transform):
    # image = Image.open(image_path).convert("RGB")
    image = read_image(image_path, ImageReadMode.RGB)
    try:
        image = transform(image, return_tensors="pt")
    except:
        pass
    return image

def add_hidden_states_to_dd(row, all_hidden_states, final_clip_embd, pooled_hidden_states):
    # pool outputs of every hidden state
    file_name = Path(row.video_name).stem
    for ind, layer in enumerate(all_hidden_states):
        key = f"attn_head_{ind}"
        pooled_layer = layer[:, 0, :].flatten().cpu().detach().numpy()
        layer_dd = {file_name: pooled_layer}
        if key in pooled_hidden_states.keys():
            pooled_hidden_states[key].update(layer_dd)
        else:
            pooled_hidden_states[key] = layer_dd

    # add the final clip embedding
    key = 'final_clip_embd'
    layer_dd = {file_name: final_clip_embd.flatten().cpu().detach().numpy()}
    if key in pooled_hidden_states.keys():
        pooled_hidden_states[key].update(layer_dd)
    else:
        pooled_hidden_states[key] = layer_dd

    return pooled_hidden_states

    # embd_save.save(embd, input_name, modality)

#%%
if __name__ == '__main__':
    """ A Script for taking a trained CLIP model weights, evaluating it on the Momments in time dataset activity\
     classification task, and extracting the embeddings of the hidden states of the model.
     To check the performance of the original CLIP model, set the config_path to None."""

    stage = 'val'
    device = 'mps' if torch.cuda.is_available() else 'cpu'
    config_path = '/Users/alonz/PycharmProjects/Clip_FineTune/clip_finetune/runs/20240627_082822_easy-snowball-132_6r1k5hlc/config.yaml'
    annot_df_path = '/Users/alonz/PycharmProjects/merlot_reserve/demo/combined_annotations.csv'
    # config_path = None
    training_name = 'original' if config_path is None else Path(config_path).parent.stem
    embedding_results_path = Path('/Users/alonz/PycharmProjects/clip2brain/features/CLIP') / training_name
    embedding_results_path.mkdir(parents=True, exist_ok=True)


    config = load_config_and_prepare_for_eval(config_path=config_path, stage=stage)
    model, processor = load_model_from_checkpoint(config['checkpoint_path'], device)
    pkg_dir = research_utils.__path__[0]
    annot_df = pd.read_csv(annot_df_path)
    config = yaml.safe_load(open(os.path.join(pkg_dir, 'model_evaluations', 'embd.yml'), 'r'))
    # embd_save = EmbeddingSave(config)
    # activities = get_activites(config, annot_df)

    annot_df = get_img_paths(annot_df)

    # get text features for all labels:
    all_labels = annot_df['label'].unique()
    # tokenize the labels
    label_mat = []
    for label in all_labels:
        label_tokens = processor.tokenizer(label, return_tensors="pt", padding=True, truncation=True)
        label_mat.append(model.get_text_features(**label_tokens.data))
    label_mat = torch.stack(label_mat, dim=1).to(device).squeeze()
    pooled_hidden_states = {}
    results_dd = {}
    results_dd['predictions'] = {}
    top1_acc = 0
    top5_acc = 0
    for ind, row in annot_df.iterrows():
        # answers = get_answers(config, row)
        # print(f"{ind}. video:{row.video_name} Answer:{answers}")
        # if '-YwZOeyAQC8_15' in row.video_name:
        path = row.path
        gt_label = row.label
        vide_name = Path(row.video_name).stem
        img = load_image(path, processor.image_processor)
        image_features, vision_model_outputs = model.get_image_features(**img.data, output_hidden_states=True)
        pooled_hidden_states = add_hidden_states_to_dd(row, vision_model_outputs.hidden_states,
                                                       image_features, pooled_hidden_states)

        top_indices = torch.matmul(label_mat, image_features.t()).squeeze().argsort(descending=True)
        pred_label = all_labels[top_indices[0]]
        top5_labels = [all_labels[i] for i in top_indices[:5]]

        if gt_label in pred_label:
            top1_acc += gt_label in pred_label
        if gt_label in top5_labels:
            top5_acc += gt_label in top5_labels
        # top5_acc /= ind+1
        # top1_acc /= ind+1
        print(f"{ind}. video:{vide_name} GT:{gt_label} Pred:{pred_label} "
              f"Top1acc:{top1_acc/(ind+1)} Top5acc:{top5_acc/(ind+1)}")
        results_dd['predictions'].update({vide_name: {'gt_label': gt_label, 'preds': top5_labels}})

results_dd.update({'top1_acc': top1_acc / annot_df.shape[0], 'top5_acc': top5_acc / annot_df.shape[0]})
# save performance results on Momments in time dataset
performance_results_path = embedding_results_path / 'performance_results.yaml'
with open(performance_results_path, 'w') as f:
    yaml.dump(results_dd, f)
print(f"Saved performance results to {performance_results_path}")
# save every layher hidden states
for layer_name, value in pooled_hidden_states.items():
    layer_path = embedding_results_path / f"{layer_name}.npy"
    np.save(layer_path, value)
print(f"Saved all hidden states to {embedding_results_path}")
# original clip: Top1acc:0.17959183673469387 Top5acc:0.6
# dazzling_lake_123 (finetuned on WW + coco humans) Top1acc:0.3224489795918367 Top5acc:0.6204081632653061
#%%
# check the saved embeddings
embd_path = "/Users/alonz/PycharmProjects/clip2brain/features/CLIP/20240618_071241_dazzling-lake-123_br35o4ls/final_clip_embd.npy"
embd = np.load(embd_path, allow_pickle=True).item()

#%%
# check the saved performance results
config = yaml.safe_load(open(performance_results_path, 'r'))
