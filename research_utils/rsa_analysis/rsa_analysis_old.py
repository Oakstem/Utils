import glob
import math
import os
import shutil
import time
from os import path, remove
import random
import matplotlib.pyplot as plt

import numpy as np
from natsort import natsorted
from data_processing import data_prep as dp
import pandas as pd
from models.models import PstsDecoder
from models import params_utils as pu
import time
import torch
from torchmetrics.functional import pairwise_cosine_similarity
from models.models import PstsDecoder
import data_processing.data_utils as du
from data_processing.dataset_types import VideoDataset, AudioDataset, \
    CombinedDataset
import neptune
from tqdm import tqdm
from models import params_utils as pu
from training.training_utils import run_one_batch_psts
import evaluate_script as es
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import pingouin as pg

from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from natsort import natsorted
from torch import Tensor, zeros, device
from torchvision import transforms
from tqdm.notebook import tqdm


def prepare_data_for_preprocessing(src_dir):
    video_dest = "stimuli_video_data"
    audio_dest = "stimuli_audio_data"
    os.makedirs(video_dest, exist_ok=True)
    os.makedirs(audio_dest, exist_ok=True)

    video_samples = sorted(
        glob.glob(path.join(src_dir, '*.mp4')))
    print(video_samples)
    audio_samples = sorted(
        glob.glob(path.join(src_dir, '*.wav')))
    for video_file, audio_file in zip(video_samples, audio_samples):
        video_file_name = video_file.split("\\")[1]
        audio_file_name = audio_file.split("\\")[1]
        id = "id_" + video_file_name.split("_")[0]
        sample_number = video_file_name.split("_")[1].split(".")[0]
        id_video_dir = os.path.join(video_dest, id)
        id_audio_dir = os.path.join(audio_dest, id)

        os.makedirs(id_video_dir, exist_ok=True)
        os.makedirs(id_audio_dir, exist_ok=True)

        # Create the sample directory inside the ID directory
        video_num_dir = os.path.join(id_video_dir, f"video_{sample_number}")
        audio_num_dir = os.path.join(id_audio_dir, f"audio_{sample_number}")
        print(video_num_dir)

        os.makedirs(video_num_dir, exist_ok=True)
        os.makedirs(audio_num_dir, exist_ok=True)

        # Move the file to the sample directory
        shutil.move(os.path.join(src_dir, video_file_name), os.path.join(video_num_dir, video_file_name))
        shutil.move(os.path.join(src_dir, audio_file_name), os.path.join(audio_num_dir, audio_file_name))

def create_vox_samples_dir_for_rsa(test_dir):
    file_path = r"dataset/test/160k_test_500ms/data_md_0.xlsx"
    df = pd.read_excel(file_path)
    destination_directory1 = 'vox_samples_rsa_60'
    destination_directory2 = 'vox_samples_rsa_30'
    small_list = []
    large_list = []

    num_samples_to_copy = 30

    # List all directories in the source directory
    all_ids = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]

    # Randomly select num_samples_to_copy directories
    selected_ids = random.sample(all_ids, num_samples_to_copy)

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_directory1, exist_ok=True)
    os.makedirs(destination_directory2, exist_ok=True)

    # Copy the selected directories and their contents to the destination directory
    for speaker_id in selected_ids:
        speaker_data = df[df['speaker_id'] == speaker_id]
        # Get unique 'num_of_video' values for the current speaker
        unique_num_values = speaker_data['id_part'].unique()

        # If there are less than 2 unique 'num_of_video' values, skip this speaker
        if len(unique_num_values) < 2:
            print("there is a speaker with one video")
            continue

        # Get the first two unique 'num_of_video' values
        num_value_1 = unique_num_values[0]
        num_value_2 = unique_num_values[1]

        # Get the sample indices corresponding to these unique 'num_of_video' values
        sample_index_1 = speaker_data[speaker_data['id_part'] == num_value_1].iloc[0]['sample_index']
        sample_index_2 = speaker_data[speaker_data['id_part'] == num_value_2].iloc[0]['sample_index']
        small_list.append(sample_index_1)
        large_list.append(sample_index_1)
        large_list.append(sample_index_2)
        dir_name1 = f"sample_{sample_index_1}"
        dir_name2 = f"sample_{sample_index_2}"
        source_path1 = os.path.join(r"dataset/test/160k_test_500ms", dir_name1)
        source_path2 = os.path.join(r"dataset/test/160k_test_500ms", dir_name2)

        destination_path1 = os.path.join(destination_directory1, dir_name1)
        destination_path2 = os.path.join(destination_directory1, dir_name2)
        destination_path3 = os.path.join(destination_directory2, dir_name1)

        shutil.copytree(source_path1, destination_path1)
        shutil.copytree(source_path2, destination_path2)
        shutil.copytree(source_path1, destination_path3)

        print(f"Copying {dir_name1} and {dir_name2} of speaker {speaker_id} to destination.")

    print("Copying completed.")
    filtered_data_30 = df[df["sample_index"].isin(small_list)]
    filtered_data_60 = df[df["sample_index"].isin(large_list)]
    filtered_file_path_30 = "vox_samples_rsa_30/data_md.xlsx"
    filtered_data_30.to_excel(filtered_file_path_30, index=False)
    filtered_file_path_60 = "vox_samples_rsa_60/data_md.xlsx"
    filtered_data_60.to_excel(filtered_file_path_60, index=False)

    dp.get_mean_and_std(destination_directory1)
    dp.get_mean_and_std(destination_directory2)

def get_model(path_to_load, num_frames=3):
    batch_size = 30
    use_end_frame = True
    use_decoder = True

    model_params = {'batch_size': batch_size,
                    'num_frames': num_frames,
                    'use_end_frame': use_end_frame,
                    'use_decoder': use_decoder,
                    'dim_resnet_to_transformer': 2048,
                    'num_heads': 4,
                    'num_layers': 2,
                    'batch_first': True,
                    'dim_feedforward': 2048,
                    # equal to dim_resnet_to_transformer
                    'num_output_features': 512,
                    'dropout': 0.3,
                    'mask': torch.triu(
                        torch.ones(num_frames + int(use_end_frame),
                                   num_frames + int(use_end_frame)), 1).bool(),
                    'seed': seed}
    video_params = pu.init_Video_decoder_params(num_frames=num_frames,
                                                dim_resnet_to_transformer=
                                                model_params[
                                                    "dim_resnet_to_transformer"],
                                                num_heads=model_params[
                                                    "num_heads"],
                                                dim_feedforward=model_params[
                                                    "dim_feedforward"],
                                                batch_first=model_params[
                                                    "batch_first"],
                                                num_layers=model_params[
                                                    "num_layers"],
                                                num_output_features=
                                                model_params[
                                                    "num_output_features"],
                                                mask=model_params["mask"],
                                                dropout=model_params["dropout"],
                                                max_len=100,
                                                use_decoder=use_decoder,
                                                use_end_frame=use_end_frame)
    audio_params = pu.init_audio_decoder_params(num_frames=num_frames,
                                                dim_resnet_to_transformer=
                                                model_params[
                                                    "dim_resnet_to_transformer"],
                                                num_heads=model_params[
                                                    "num_heads"],
                                                dim_feedforward=model_params[
                                                    "dim_feedforward"],
                                                batch_first=model_params[
                                                    "batch_first"],
                                                num_layers=model_params[
                                                    "num_layers"],
                                                num_output_features=
                                                model_params[
                                                    "num_output_features"],
                                                mask=model_params["mask"],
                                                dropout=model_params["dropout"],
                                                max_len=100,
                                                use_decoder=use_decoder,
                                                use_end_frame=use_end_frame)

    psts_params = pu.init_psts_decoder_params(num_frames=num_frames,
                                              video_params=video_params,
                                              audio_params=audio_params)
    psts_encoder = PstsDecoder(psts_params, False, use_end_frame, use_decoder)
    psts_encoder.load_model(path_to_load)
    return psts_encoder

def get_psts_representation(model, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    num_of_samples = len(dataset)
    outputs = [[None, None, None] for _ in range(num_of_samples)]

    for i in tqdm(range(num_of_samples)):
        v1, a1, l1 = dataset[i]
        encode_v1, encode_a1 = es.fast_run_and_compare(model, v1, a1, device)
        outputs[i] = [encode_v1, encode_a1, l1]
    return outputs

def get_face_model_embeddings(src_dir):
    pass

def get_audio_model_embedding(audio_model, audio_sample):
    """
    audio_sample: a wav file load by torchaudio.load (take the first argument)
    """
    return audio_model.encode_batch(audio_sample)

def get_audio_model_representations(src_dir):
    # for fmri - data_md_0, else - data_md
    xlsx_path = path.join(src_dir, "data_md_0.xlsx")
    md_df = pd.read_excel(xlsx_path)
    representations = []
    speaker_verification_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb")

    samples_directories = natsorted(glob.glob(path.join(src_dir, "sample*")))
    for sample_directory in samples_directories:
        sample_representations = []
        sample_wav_files = natsorted(glob.glob(path.join(sample_directory, 'audio', '*.wav')))
        wav_files_for_embedding = sample_wav_files[1:4]
        sample_index = int(sample_directory.split("sample_")[-1])
        row = md_df[md_df['sample_index'] == sample_index]
        speaker_id = row.iloc[0]['speaker_id']
        for wav_file in wav_files_for_embedding:
            sample_representations.append(get_audio_model_embedding(speaker_verification_model,
                                                                    torchaudio.load(wav_file)[0]))
        # End frame
        sample_representations.append(sum(sample_representations)/3)
        sample_representations.append(speaker_id)
        representations.append(sample_representations)
    return representations

def create_and_save_rdm(embeddings, labels, dest_path):
    rdm = np.zeros((len(embeddings), len(embeddings)))
    for i, first_em in enumerate(embeddings):
        for j, second_em in enumerate(embeddings):
            rdm[i, j] = pairwise_cosine_similarity(first_em, second_em)

    n = len(labels)
    rdm_df = pd.DataFrame(data=rdm, columns=labels, index=labels)
    rdm_df.to_excel(dest_path)

    print("rdm saved successfuly")

def create_audio_model_rdms(save_dir, audio_model_dir):
    # Create whole audio (audio model) rdm
    whole_audio_embeddings = []
    whole_labels = []
    audio_frames_embeddings = []
    audio_frames_labels = []
    save_whole_audio_rdm_path = path.join(save_dir, "audio_model_whole_audio_rdm_30.xlsx")
    save_audio_frames_rdm_path = path.join(save_dir, "audio_model_audio_frames_rdm_30.xlsx")
    audio_model_representations = get_audio_model_representations(audio_model_dir)
    for representation in audio_model_representations:
        whole_audio_embeddings.append(representation[-2][0])
        audio_frames_embeddings.append(representation[0][0])
        audio_frames_embeddings.append(representation[1][0])
        audio_frames_embeddings.append(representation[2][0])
        whole_labels.append(representation[-1])
        audio_frames_labels.extend([representation[-1]] * 3)
    print("creating whole audio - audio model - rdm")
    create_and_save_rdm(whole_audio_embeddings, whole_labels, save_whole_audio_rdm_path)
    print("creating audio frames - audio model - rdm")
    create_and_save_rdm(audio_frames_embeddings, audio_frames_labels, save_audio_frames_rdm_path)


def create_psts_rdms(model, data_dir, save_dir, fmri=False, fmri_representations=None, type_fmri=None):
    if not fmri:
        dataset = es.get_dataset(data_dir)
        representations = get_psts_representation(model, dataset)
        save_whole_audio_rdm_path = path.join(save_dir, "psts_model_whole_audio_rdm_60.xlsx")
        save_audio_frames_rdm_path = path.join(save_dir, "psts_model_audio_frames_rdm_60.xlsx")
        save_whole_video_rdm_path = path.join(save_dir, "psts_model_whole_video_rdm_60.xlsx")
        save_video_frames_rdm_path = path.join(save_dir, "psts_model_video_frames_rdm_60.xlsx")
    else:
        representations = fmri_representations
        save_whole_audio_rdm_path = path.join(save_dir, f"fmri_{type_fmri}_whole_audio_rdm.xlsx")
        save_audio_frames_rdm_path = path.join(save_dir, f"fmri_{type_fmri}_audio_frames_rdm.xlsx")
        save_whole_video_rdm_path = path.join(save_dir, f"fmri_{type_fmri}_whole_video_rdm.xlsx")
        save_video_frames_rdm_path = path.join(save_dir, f"fmri_{type_fmri}_video_frames_rdm.xlsx")

    video_end_frames_embeddings = []
    audio_end_frames_embeddings = []
    end_frames_labels = []
    audio_frames_embeddings = []
    video_frames_embeddings = []
    frames_labels = []

    for representation in representations:
        video_frames_embeddings.append(representation[0][0][0].unsqueeze(0))
        video_frames_embeddings.append(representation[0][0][1].unsqueeze(0))
        video_frames_embeddings.append(representation[0][0][2].unsqueeze(0))
        video_end_frames_embeddings.append(representation[0][0][3].unsqueeze(0))
        audio_frames_embeddings.append(representation[1][0][0].unsqueeze(0))
        audio_frames_embeddings.append(representation[1][0][1].unsqueeze(0))
        audio_frames_embeddings.append(representation[1][0][2].unsqueeze(0))
        audio_end_frames_embeddings.append(representation[1][0][3].unsqueeze(0))
        end_frames_labels.append(representation[-1])
        frames_labels.extend([representation[-1]] * 3)
    print("creating whole audio rdm")
    create_and_save_rdm(audio_end_frames_embeddings, end_frames_labels, save_whole_audio_rdm_path)
    print("creating audio frames rdm")
    create_and_save_rdm(audio_frames_embeddings, frames_labels, save_audio_frames_rdm_path)
    print("creating whole video rdm")
    create_and_save_rdm(video_end_frames_embeddings, end_frames_labels, save_whole_video_rdm_path)
    print("creating video frames rdm")
    create_and_save_rdm(video_frames_embeddings, frames_labels, save_video_frames_rdm_path)

def create_fmri_psts_rdms(fmri_first_dir, fmri_second_dir, model, save_dir):

    dataset_first = es.get_dataset(fmri_first_dir)
    representations_first = get_psts_representation(model, dataset_first)
    dataset_second = es.get_dataset(fmri_second_dir)
    representations_second = get_psts_representation(model, dataset_second)
    mean_representations_video_first = []
    mean_representations_video_second = []
    mean_representations_audio_first = []
    mean_representations_audio_second = []
    for i in range(0, len(representations_first), 3):
        video_element1_tensors_first = representations_first[i][0]
        video_element2_tensors_first = representations_first[i+1][0]
        video_element3_tensors_first = representations_first[i+2][0]
        audio_element1_tensors_first = representations_first[i][1]
        audio_element2_tensors_first = representations_first[i+1][1]
        audio_element3_tensors_first = representations_first[i+2][1]
        label_first = representations_first[i][2]
        video_element1_tensors_second = representations_second[i][0]
        video_element2_tensors_second = representations_second[i + 1][0]
        video_element3_tensors_second = representations_second[i + 2][0]
        audio_element1_tensors_second = representations_second[i][1]
        audio_element2_tensors_second = representations_second[i + 1][1]
        audio_element3_tensors_second = representations_second[i + 2][1]
        label_second = representations_second[i][2]

        mean_video_element_first = (video_element1_tensors_first + video_element2_tensors_first
                                    + video_element3_tensors_first) / 3
        mean_audio_element_first = (audio_element1_tensors_first + audio_element2_tensors_first
                                    + audio_element3_tensors_first) / 3

        mean_video_element_second = (video_element1_tensors_second + video_element2_tensors_second
                                    + video_element3_tensors_second) / 3
        mean_audio_element_second = (audio_element1_tensors_second + audio_element2_tensors_second
                                    + audio_element3_tensors_second) / 3

        mean_representations_video_first.append(
            [mean_video_element_first[0][3], label_first])

        mean_representations_video_second.append(
            [mean_video_element_second[0][3], label_second])

        mean_representations_audio_first.append(
            [mean_audio_element_first[0][3], label_first])

        mean_representations_audio_second.append(
            [mean_audio_element_second[0][3], label_second])

    create_fmri_rdms(save_dir, mean_representations_video_first, mean_representations_video_second, "psts_video")
    create_fmri_rdms(save_dir, mean_representations_audio_first, mean_representations_audio_second, "psts_audio")


def create_fmri_rdms(save_dir, first_representations, second_representations, model_type):
    save_rdm_path = path.join(save_dir, f"fmri_{model_type}_rdm.xlsx")

    first_embeddings = []
    second_embeddings = []
    first_labels = []
    second_labels = []

    for representation in first_representations:
        first_embeddings.append(representation[0].unsqueeze(0))
        first_labels.append(representation[1])
    for representation in second_representations:
        second_embeddings.append(representation[0].unsqueeze(0))
        second_labels.append(representation[1])

    print(f"creating {model_type} fmri rdm")
    create_and_save_fmri_rdm(first_embeddings, second_embeddings, first_labels, second_labels, save_rdm_path)

def create_and_save_fmri_rdm(first_embeddings, second_embeddings, first_labels, second_labels, dest_path):
    rdm = np.zeros((len(first_embeddings), len(second_embeddings)))
    for i, first_em in enumerate(first_embeddings):
        for j, second_em in enumerate(second_embeddings):
            rdm[i, j] = pairwise_cosine_similarity(first_em, second_em)

    n = len(first_labels)
    rdm_df = pd.DataFrame(data=rdm, columns=first_labels, index=second_labels)
    rdm_df.to_excel(dest_path)
    print("rdm saved successfuly")

def create_audio_model_fmri_rdms(save_dir, fmri_first_dir, fmri_second_dir):
    first_audios_embeddings = []
    first_labels = []
    second_audios_embeddings = []
    second_labels = []
    save_audio_fmri_rdm_path = path.join(save_dir, "audio_model_fmri_rdm.xlsx")
    first_audios_representations = get_audio_model_representations(fmri_first_dir)
    print(first_audios_representations)
    second_audios_representations = get_audio_model_representations(fmri_second_dir)

    for i in range(0, len(first_audios_representations), 3):
        first_audios_embeddings.append((first_audios_representations[i][-2][0] +
                                        first_audios_representations[i+1][-2][0] +
                                        first_audios_representations[i+2][-2][0]) / 3)
        first_labels.append(first_audios_representations[i][-1])
    for i in range(0, len(second_audios_representations), 3):
        second_audios_embeddings.append((second_audios_representations[i][-2][0] +
                                        second_audios_representations[i+1][-2][0] +
                                        second_audios_representations[i+2][-2][0]) / 3)
        second_labels.append(second_audios_representations[i][-1])

    print("creating audio model fmri rdm")
    create_and_save_fmri_rdm(first_audios_embeddings, second_audios_embeddings, first_labels, second_labels,
                             save_audio_fmri_rdm_path)


def create_video_model_fmri_rdms(save_dir, fmri_first_dir, fmri_second_dir):
    first_videos_embeddings = []
    first_labels = []
    second_videos_embeddings = []
    second_labels = []
    save_videos_fmri_rdm_path = path.join(save_dir, "video_model_fmri_rdm.xlsx")
    first_videos_representations = get_face_recognition_embeddings(fmri_first_dir)
    print(first_videos_representations)
    second_videos_representations = get_face_recognition_embeddings(fmri_second_dir)

    for i in range(0, len(first_videos_representations), 3):
        first_videos_embeddings.append((first_videos_representations[i][-2] +
                                        first_videos_representations[i+1][-2] +
                                        first_videos_representations[i+2][-2]) / 3)
        first_labels.append(first_videos_representations[i][-1])
    for i in range(0, len(second_videos_representations), 3):
        second_videos_embeddings.append((second_videos_representations[i][-2] +
                                        second_videos_representations[i+1][-2] +
                                        second_videos_representations[i+2][-2]) / 3)
        second_labels.append(second_videos_representations[i][-1])

    print("creating video model fmri rdm")
    create_and_save_fmri_rdm(first_videos_embeddings, second_videos_embeddings, first_labels, second_labels,
                             save_videos_fmri_rdm_path)


def get_video_embedding(video_imgs_dir: str, mtcnn, model, id):
    normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
    im_paths =[ip for ip in natsorted(glob.glob(path.join(video_imgs_dir,
                                                     '*.jpg')))]
    output, outputs = [], []
    for im_path in im_paths:
        img = Image.open(im_path)
        if img.mode != 'RGB':  # PNG imgs are RGBA
            img = img.convert('RGB')
        img = mtcnn(img)
        if img is not None:
            img = img.cuda()
            img = img / 255.0
            img = normalize_imagenet(img).to(device('cuda:1')).unsqueeze(
                0).float()
            output.append(model(img).cpu())
        else:
                print(f'failed mtcnn in {im_path}')

        if len(output) == 3: #add mean frame
            stacked_frames_embedding = torch.stack(output)
            output.append(torch.mean(stacked_frames_embedding, dim=0))
            output.append(id)
            outputs.append(output)
            output = []
    return outputs


def get_all_video_embeddings(videos_dir:str, mtcnn, model):
    all_video_embeddings = []
    video_dirs = [subdir for subdir in glob.glob(os.path.join(videos_dir, '*'))
                  if os.path.isdir(subdir)]
    for video_dir in tqdm(video_dirs):
        id = os.path.basename(video_dir)
        embeddings = get_video_embedding(video_dir, mtcnn, model, id)
        all_video_embeddings.extend(embeddings)
    return all_video_embeddings


def get_face_recognition_embeddings(videos_dir: str):
    mtcnn = MTCNN(image_size=160, post_process=False,
                  device='cuda:0')  # .cuda()
    model = InceptionResnetV1(pretrained='vggface2', device='cuda:1').eval()
    return get_all_video_embeddings(videos_dir, mtcnn, model)

def create_video_model_rdms(save_dir, video_model_dir):
    whole_video_embeddings = []
    whole_labels = []
    video_frames_embeddings = []
    video_frames_labels = []
    save_whole_video_rdm_path = path.join(save_dir, "video_model_whole_video_rdm_60.xlsx")
    save_video_frames_rdm_path = path.join(save_dir, "video_model_video_frames_rdm_60.xlsx")
    video_model_representations = get_face_recognition_embeddings(video_model_dir)
    for representation in video_model_representations:
        whole_video_embeddings.append(representation[-2])
        video_frames_embeddings.append(representation[0])
        video_frames_embeddings.append(representation[1])
        video_frames_embeddings.append(representation[2])
        whole_labels.append(representation[-1])
        video_frames_labels.extend([representation[-1]] * 3)
    print("creating whole video - video model - rdm")
    create_and_save_rdm(whole_video_embeddings, whole_labels, save_whole_video_rdm_path)
    print("creating video frames - video model - rdm")
    create_and_save_rdm(video_frames_embeddings, video_frames_labels, save_video_frames_rdm_path)

def create_rsa_from_two_rdms(path_to_first_rdm, path_to_second_rdm
                             ,first_rdm_type, second_rdm_type, rsa_type,
                             file_to_save_full_corr_rsa, file_to_save_partial_corr_rsa):
    rdm1 = pd.read_excel(path_to_first_rdm, index_col=0).values
    rdm2 = pd.read_excel(path_to_second_rdm, index_col=0).values
    n = rdm1.shape[0]
    indices = np.triu_indices(n, k=-1)  # Exclude main diagonal
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
        full_corr_data = pd.read_excel(file_to_save_full_corr_rsa)
        partial_corr_data = pd.read_excel(file_to_save_partial_corr_rsa)

    except FileNotFoundError:
        # Create a new DataFrame if the file doesn't exist
        full_corr_data = pd.DataFrame()
        partial_corr_data = pd.DataFrame()

    # Concatenate the existing data and new results
    combined_full_corr_data = pd.concat([full_corr_data, full_corr_results], ignore_index=True)
    combined_partial_corr_data = pd.concat([partial_corr_data, partial_corr_result], ignore_index=True)

    # Save the combined data to a new sheet in the Excel file
    combined_full_corr_data.to_excel(file_to_save_full_corr_rsa, index=False)
    combined_partial_corr_data.to_excel(file_to_save_partial_corr_rsa, index=False)


def plot_rdm(rsa_path):
    correlation_df = pd.read_excel(rsa_path, index_col=0)
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

def split_stimuli_processed(data_dir):
    dest_dir1 = "stimuli_first"
    dest_dir2 = "stimuli_second"
    samples_directories = natsorted(glob.glob(path.join(data_dir, "sample*")))
    md_file = path.join(data_dir, "data_md_0.xlsx")

    os.makedirs(dest_dir1, exist_ok=True)
    os.makedirs(dest_dir2, exist_ok=True)

    counter = 0
    id = 0
    for sample_directory in samples_directories:
        dir_name = f"sample_{counter + id}"
        if counter <= 2:
            destination_path = os.path.join(dest_dir1, dir_name)
            shutil.copytree(sample_directory, destination_path)
        else:
            destination_path = os.path.join(dest_dir2, dir_name)
            shutil.copytree(sample_directory, destination_path)
        counter += 1
        if counter > 5:
            counter = 0
            id += 6
    dest_file1 = path.join(dest_dir1, "data_md_0.xlsx")
    dest_file2 = path.join(dest_dir2, "data_md_0.xlsx")
    shutil.copyfile(md_file, dest_file1)
    shutil.copyfile(md_file, dest_file2)

def script_for_all_rsas():
    create_rsa_from_two_rdms("rsa_results/psts_model_audio_frames_rdm_30.xlsx",
                             "rsa_results/audio_model_audio_frames_rdm_30.xlsx",
                             "psts_audio_frames_30", "audio_model_frames_30", "psts_audio_model_frames_30_rsa",
                             "rsa_results/rsa_full_corr_values.xlsx", "rsa_results/rsa_partial_corr_values.xlsx")

    create_rsa_from_two_rdms("rsa_results/psts_model_audio_frames_rdm_60.xlsx",
                             "rsa_results/audio_model_audio_frames_rdm_60.xlsx",
                             "psts_audio_frames_60", "audio_model_frames_60", "psts_audio_model_frames_60_rsa",
                             "rsa_results/rsa_full_corr_values.xlsx", "rsa_results/rsa_partial_corr_values.xlsx")

    create_rsa_from_two_rdms("rsa_results/psts_model_whole_audio_rdm_30.xlsx",
                             "rsa_results/audio_model_whole_audio_rdm_30.xlsx",
                             "psts_whole_audio_30", "audio_model_whole_audio_30", "psts_audio_model_whole_audio_30_rsa",
                             "rsa_results/rsa_full_corr_values.xlsx", "rsa_results/rsa_partial_corr_values.xlsx")

    create_rsa_from_two_rdms("rsa_results/psts_model_whole_audio_rdm_60.xlsx",
                             "rsa_results/audio_model_whole_audio_rdm_60.xlsx",
                             "psts_whole_audio_60", "audio_model_whole_audio_60", "psts_audio_model_whole_audio_60_rsa",
                             "rsa_results/rsa_full_corr_values.xlsx", "rsa_results/rsa_partial_corr_values.xlsx")

    create_rsa_from_two_rdms("rsa_results/psts_model_video_frames_rdm_30.xlsx",
                             "rsa_results/video_model_video_frames_rdm_30.xlsx",
                             "psts_video_frames_30", "video_model_frames_30", "psts_video_model_frames_30_rsa",
                             "rsa_results/rsa_full_corr_values.xlsx", "rsa_results/rsa_partial_corr_values.xlsx")

    create_rsa_from_two_rdms("rsa_results/psts_model_video_frames_rdm_60.xlsx",
                             "rsa_results/video_model_video_frames_rdm_60.xlsx",
                             "psts_video_frames_60", "video_model_frames_60", "psts_video_model_frames_60_rsa",
                             "rsa_results/rsa_full_corr_values.xlsx", "rsa_results/rsa_partial_corr_values.xlsx")

    create_rsa_from_two_rdms("rsa_results/psts_model_whole_video_rdm_30.xlsx",
                             "rsa_results/video_model_whole_video_rdm_30.xlsx",
                             "psts_whole_video_30", "video_model_whole_video_30", "psts_video_model_whole_video_30_rsa",
                             "rsa_results/rsa_full_corr_values.xlsx", "rsa_results/rsa_partial_corr_values.xlsx")

    create_rsa_from_two_rdms("rsa_results/psts_model_whole_video_rdm_60.xlsx",
                             "rsa_results/video_model_whole_video_rdm_60.xlsx",
                             "psts_whole_video_60", "video_model_whole_video_60", "psts_video_model_whole_video_60_rsa",
                             "rsa_results/rsa_full_corr_values.xlsx", "rsa_results/rsa_partial_corr_values.xlsx")
    #fmri
    create_rsa_from_two_rdms("rsa_results/fmri_first_audio_frames_rdm.xlsx",
                             "rsa_results/fmri_second_audio_frames_rdm.xlsx",
                             "psts_first_fmri_audio_frames", "psts_second_fmri_audio_frames", "psts_fmri_audio_frames_rsa",
                             "rsa_results/rsa_full_corr_values.xlsx", "rsa_results/rsa_partial_corr_values.xlsx")

    create_rsa_from_two_rdms("rsa_results/fmri_first_whole_audio_rdm.xlsx",
                             "rsa_results/fmri_second_whole_audio_rdm.xlsx",
                             "psts_first_fmri_whole_audio", "psts_second_fmri_whole_audio", "psts_fmri_whole_audio_rsa",
                             "rsa_results/rsa_full_corr_values.xlsx", "rsa_results/rsa_partial_corr_values.xlsx")

    create_rsa_from_two_rdms("rsa_results/fmri_first_video_frames_rdm.xlsx",
                             "rsa_results/fmri_second_video_frames_rdm.xlsx",
                             "psts_first_fmri_video_frames", "psts_second_fmri_video_frames", "psts_fmri_video_frames_rsa",
                             "rsa_results/rsa_full_corr_values.xlsx", "rsa_results/rsa_partial_corr_values.xlsx")

    create_rsa_from_two_rdms("rsa_results/fmri_first_whole_video_rdm.xlsx",
                             "rsa_results/fmri_second_whole_video_rdm.xlsx",
                             "psts_first_fmri_whole_video", "psts_second_fmri_whole_video", "psts_fmri_whole_video_rsa",
                             "rsa_results/rsa_full_corr_values.xlsx", "rsa_results/rsa_partial_corr_values.xlsx")



if __name__ == '__main__':
    #prepare_data_for_preprocessing("stimuli")
    #create_vox_samples_dir_for_rsa("dataset/test/voxceleb_video_test/mp4")

    seed = 42
    torch.manual_seed(seed)
    #data_dir = r'vox_samples_rsa_60'
    best_model_dir = r'models/check transformer whole DS, no gradient BS= 54, num frames=3, end_frame=True, LR= 0.0000001, drop=0.3, dim_feedforward=2048, num_outputfeature=512, train=0.9, num_heads=4, num_layers=2/best_model'
    #model = get_model(best_model_dir)
    #create_audio_model_rdms("rsa_results", data_dir)
    #create_psts_rdms(model, data_dir, "rsa_results")
    #speaker_verification_model = EncoderClassifier.from_hparams(
      # source="speechbrain/spkrec-ecapa-voxceleb")
    #audio_rep = get_audio_model_embedding(speaker_verification_model, torchaudio.load("sample_13877_a_11.wav")[0])
    #print(audio_rep[-1][0].size())
    create_rsa_from_two_rdms("rpSTS_Faces_mean.xlsx", "fmri_psts_audio_rdm.xlsx",
                           "psts_audio_fmri", "psts_video_fmri", "psts_video_fmri_real_rsa",
                       "rsa_results/rsa_full_corr_values.xlsx", "rsa_results/rsa_partial_corr_values.xlsx")
    #plot_rdm("fmri_first_video_frames_rdm.xlsx")
    #neptune = neptune.init_run(
     #   project="psts-through-unsupervised-learning/psts",
      #  api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzODRhM2YzNi03Nzk4LTRkZDctOTJiZS1mYjMzY2EzMDMzOTMifQ==")
    #print(get_psts_representation(model, dataset))
    #split_stimuli_processed("stimuli_processed")
    #create_fmri_rdms("stimuli_second", model, "rsa_results", "second")
    #print(get_face_recognition_embeddings("face_model_data"))
    #create_video_model_rdms("rsa_results", "face_model_data_60")
    #script_for_all_rsas()
    #create_fmri_psts_rdms("stimuli_first", "stimuli_second", model, "rsa_results")
    #create_audio_model_fmri_rdms("rsa_results", "stimuli_first", "stimuli_second")
    #create_video_model_fmri_rdms("rsa_results", "stimuli_first_face_model", "stimuli_second_face_model")
