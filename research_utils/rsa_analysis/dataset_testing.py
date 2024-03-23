"""
Demo for doing interesting things with a video
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.append('../')

from mreserve.preprocess import video_to_segments, preprocess_video, encoder, MASK, video_to_segments_zero_shot
from mreserve.modeling import PretrainedMerlotReserve
from research_utils.rsa_analysis.embed_save import EmbeddingSave
import jax
import jax.numpy as jnp

# This handles loading the model and getting the checkpoints.
grid_size = (18, 32)
model = PretrainedMerlotReserve.from_pretrained(model_name='large', image_grid_size=grid_size)

top1_acc = []
top5_acc = []
top10_acc = []

def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

def get_times(video_file, time_interval=1.):
    video_length = get_length(video_file)

    times = []
    num_chunk = int(video_length // time_interval)
    cut_length = num_chunk * time_interval
    st = round((video_length - cut_length) / 2, 2)
    for i in range(num_chunk):
        et = st + time_interval
        times.append({'start_time': st, 'end_time': et, 'mid_time': round((st + et) / 2.0, 2)})
        st += time_interval
    return times

def concat_arrays(input_arr, key):
    res_arr = []
    for val in input_arr:
        val_to_concat = val[key]
        if len(val_to_concat.shape) < 4:
            val_to_concat = np.expand_dims(val_to_concat, axis=0)
        res_arr.append(val_to_concat)
    res_arr = np.concatenate(res_arr, axis=0)
    return res_arr

def inference(config, path, txt, options, embd_save, answer):
    try:
        filename = Path(path).stem
        times = get_times(path, time_interval=config['segment_length'])
        video_segments = video_to_segments_zero_shot(path, times=times, time_interval=config['segment_length'])
        # video_segments = video_to_segments(path)
        # video_segments = video_segments[3:11]

        # Set up a fake classification task.
        # video_segments[0]['text'] = '<|MASK|>'
        if config['multimodal']:
            video_segments[0]['text'] = txt
            video_segments[0]['use_text_as_input'] = True
            for i in range(1,len(video_segments)):
                video_segments[i]['use_text_as_input'] = False
        else:
            for i in range(0,len(video_segments)):
                video_segments[i]['use_text_as_input'] = False
                # video_segments[i]['text'] = '<|MASK|>'
                # video_segments[i]['text'] = ''

        video_pre = preprocess_video(video_segments, output_grid_size=grid_size, verbose=True)

        # Now we embed the entire video and extract the text. result is  [seq_len, H]. we extract a hidden state for every
        # MASK token
        video_pre['use_audio'] = config['use_audio_input']
        video_pre['use_video'] = config['use_video_input']
        # model.use_audio_input = config['use_audio_input']
        out_h, vis_embd, audio_embd = model.embed_video(**video_pre)
        if config['multimodal']:
            out_h_token = out_h[video_pre['tokens'] == MASK]
            embd_save(out_h_token, input_name=filename, modality='combined')
        else:
            embd_save(out_h, input_name=filename, modality='combined')

        # the following is all the labels from activitynet. why not! some of them don't make sense grammatically though.
        # options += ['Applying sunscreen', 'Archery', 'Arm wrestling', 'Assembling bicycle', 'BMX', 'Baking cookies', 'Ballet', 'Bathing dog', 'Baton twirling', 'Beach soccer', 'Beer pong', 'Belly dance', 'Blow-drying hair', 'Blowing leaves', 'Braiding hair', 'Breakdancing', 'Brushing hair', 'Brushing teeth', 'Building sandcastles', 'Bullfighting', 'Bungee jumping', 'Calf roping', 'Camel ride', 'Canoeing', 'Capoeira', 'Carving jack-o-lanterns', 'Changing car wheel', 'Cheerleading', 'Chopping wood', 'Clean and jerk', 'Cleaning shoes', 'Cleaning sink', 'Cleaning windows', 'Clipping cat claws', 'Cricket', 'Croquet', 'Cumbia', 'Curling', 'Cutting the grass', 'Decorating the Christmas tree', 'Disc dog', 'Discus throw', 'Dodgeball', 'Doing a powerbomb', 'Doing crunches', 'Doing fencing', 'Doing karate', 'Doing kickboxing', 'Doing motocross', 'Doing nails', 'Doing step aerobics', 'Drinking beer', 'Drinking coffee', 'Drum corps', 'Elliptical trainer', 'Fixing bicycle', 'Fixing the roof', 'Fun sliding down', 'Futsal', 'Gargling mouthwash', 'Getting a haircut', 'Getting a piercing', 'Getting a tattoo', 'Grooming dog', 'Grooming horse', 'Hammer throw', 'Hand car wash', 'Hand washing clothes', 'Hanging wallpaper', 'Having an ice cream', 'High jump', 'Hitting a pinata', 'Hopscotch', 'Horseback riding', 'Hula hoop', 'Hurling', 'Ice fishing', 'Installing carpet', 'Ironing clothes', 'Javelin throw', 'Kayaking', 'Kite flying', 'Kneeling', 'Knitting', 'Laying tile', 'Layup drill in basketball', 'Long jump', 'Longboarding', 'Making a cake', 'Making a lemonade', 'Making a sandwich', 'Making an omelette', 'Mixing drinks', 'Mooping floor', 'Mowing the lawn', 'Paintball', 'Painting', 'Painting fence', 'Painting furniture', 'Peeling potatoes', 'Ping-pong', 'Plastering', 'Plataform diving', 'Playing accordion', 'Playing badminton', 'Playing bagpipes', 'Playing beach volleyball', 'Playing blackjack', 'Playing congas', 'Playing drums', 'Playing field hockey', 'Playing flauta', 'Playing guitarra', 'Playing harmonica', 'Playing ice hockey', 'Playing kickball', 'Playing lacrosse', 'Playing piano', 'Playing polo', 'Playing pool', 'Playing racquetball', 'Playing rubik cube', 'Playing saxophone', 'Playing squash', 'Playing ten pins', 'Playing violin', 'Playing water polo', 'Pole vault', 'Polishing forniture', 'Polishing shoes', 'Powerbocking', 'Preparing pasta', 'Preparing salad', 'Putting in contact lenses', 'Putting on makeup', 'Putting on shoes', 'Rafting', 'Raking leaves', 'Removing curlers', 'Removing ice from car', 'Riding bumper cars', 'River tubing', 'Rock climbing', 'Rock-paper-scissors', 'Rollerblading', 'Roof shingle removal', 'Rope skipping', 'Running a marathon', 'Sailing', 'Scuba diving', 'Sharpening knives', 'Shaving', 'Shaving legs', 'Shot put', 'Shoveling snow', 'Shuffleboard', 'Skateboarding', 'Skiing', 'Slacklining', 'Smoking a cigarette', 'Smoking hookah', 'Snatch', 'Snow tubing', 'Snowboarding', 'Spinning', 'Spread mulch', 'Springboard diving', 'Starting a campfire', 'Sumo', 'Surfing', 'Swimming', 'Swinging at the playground', 'Table soccer', 'Tai chi', 'Tango', 'Tennis serve with ball bouncing', 'Throwing darts', 'Trimming branches or hedges', 'Triple jump', 'Tug of war', 'Tumbling', 'Using parallel bars', 'Using the balance beam', 'Using the monkey bar', 'Using the pommel horse', 'Using the rowing machine', 'Using uneven bars', 'Vacuuming floor', 'Volleyball', 'Wakeboarding', 'Walking the dog', 'Washing dishes', 'Washing face', 'Washing hands', 'Waterskiing', 'Waxing skis', 'Welding', 'Windsurfing', 'Wrapping presents', 'Zumba']
        label_space = model.get_label_space(options)
        if len(answer) > 0:
            embd_save(label_space[options == answer], input_name=filename, modality='text')

        # Single modality extraction
        # vis_frames = concat_arrays(video_segments, 'frame')
        # audio_frames = concat_arrays(video_segments, 'spectrogram')

        # Extract only visual embedding
        embd_save(vis_embd, input_name=filename, modality='visual')

        # Extract only audio embedding
        embd_save(audio_embd, input_name=filename, modality='audio')

        # Dot product the <|MASK|> tokens and the options together
        logits = 100.0 * jnp.einsum('bh,lh->bl', out_h_token, label_space)
        idx = jnp.argsort(-logits)

        top_1_predict = [options[idx[0, 0]]]
        top_5_predict = [options[idx[0, i]] for i in range(5)]
        top_10_predict = [options[idx[0, i]] for i in range(10)]
        print(f'Top 5 Predictions:{top_5_predict}')

        top1_acc.append(int(answer in top_1_predict))
        top5_acc.append(int(answer in top_5_predict))
        top10_acc.append(int(answer in top_10_predict))
    except:
        print(f"Bad Input for video:{filename}, skipping")

def get_answers(config, row):
    if config.get('social', False):
        answers = get_social_answers(row, th=0.7)
    else:
        answers = row.label
    return answers

def get_social_answers(row, th=0.7):
    # social_label = ['agent distance', 'facingness', 'joint action', 'communication',
    #  'cooperation', 'valence', 'arousal']
    answers = {'communicating' if row['communication'] > th else 'not communicating'}
    answers.update({'not communicating' if row['communication'] < 1-th else ''})
    answers.update({'facing each other' if row['facingness'] > th else ''})
    answers.update({'facing away from each other' if row['facingness'] < 1-th else ''})
    answers.update({'physically far from each other' if row['agent distance'] > th else ''})
    answers.update({'physically close to each other' if row['agent distance'] < 1-th else ''})
    answers.update({'acting jointly' if row['joint action'] > th else ''})
    answers.update({'acting independently' if row['joint action'] < 1-th else ''})
    answers.update({'engaging in a pleasant activity' if row['valence'] > th else ''})
    answers.update({'engaging in an unpleasant activity' if row['valence'] < 1-th else ''})
    answers.update({'engaging in an intense activity' if row['arousal'] > th else ''})
    answers.update({'engaging in a calm activity' if row['arousal'] < 1-th else ''})

    if '' in answers:
        answers.remove('')

    return answers

def get_activites(config, annot_df):
    if config.get('social', False):
        activities = ['communicating', 'not communicating', 'facing each other', 'facing away from each other',
                      'physically close to each other', 'physically far from each other', 'acting independently',
                      'acting jointly', 'engaging in a pleasant activity',
                      'engaging in an unpleasant activity', 'engaging in a calm activity',
                      'engaging in an intense activity'
                      ]
    else:
        activities = annot_df.label.unique()
    return activities

if __name__ == '__main__':
    annot_df_path = '/Users/alonz/PycharmProjects/merlot_reserve/demo/combined_annotations.csv'
    annot_df = pd.read_csv(annot_df_path)
    config = yaml.safe_load(open('embd.yml', 'r'))
    embd_save = EmbeddingSave(config)
    activities = get_activites(config, annot_df)

    prompt = 'the people in the video are<|MASK|>'

    for ind, row in annot_df.iterrows():
        answers = get_answers(config, row)
        print(f"{ind}. video:{row.video_name} Answer:{answers}")
        if '-YwZOeyAQC8_15' in row.video_name:
            inference(config=config, path=row.path, txt=prompt, options=activities, embd_save=embd_save, answer=answers)


    print(f'top1-acc: {round(sum(top1_acc) / len(top1_acc) * 100., 3)}')
    print(f'top5-acc: {round(sum(top5_acc) / len(top5_acc) * 100., 3)}')
    print(f'top10-acc: {round(sum(top10_acc) / len(top10_acc) * 100., 3)}')

    json.dump({'top1-acc': top1_acc, 'top5-acc': top5_acc, 'top10-acc': top10_acc}, open(f'embeddings/{config["save_dir"]}/acc_results.json', 'w'))