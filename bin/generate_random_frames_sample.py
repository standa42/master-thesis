import os
import cv2
import uuid
import zipfile
import random
from pathlib import Path
from random import choice
from shutil import copyfile

from config.Config import Config
from src.helpers.helper_functions import *

from src.data.video.video import Video
from src.data.video.video_pair import Video_pair
from src.data.video.video_dataset import Video_dataset

# TODO ensure none repetition of sampling

if __name__ == "__main__":
    print("Script started")
    # get all videos
    dataset = Video_dataset()
    all_pairs = dataset.get_all_video_pairs()

    # ensure directiories are made
    samples_folder = Config.DataPaths.RandomFrameSampleFolder
    samples_folder_train = samples_folder + 'train/'
    samples_folder_val = samples_folder + 'val/'
    samples_folder_test = samples_folder + 'test/'
    safe_mkdir(Config.DataPaths.RandomFrameSampleFolder)
    safe_mkdir_clean(samples_folder_train)
    safe_mkdir_clean(samples_folder_val)
    safe_mkdir_clean(samples_folder_test)
    
    # set counts to generate
    train = 1500
    val = 400
    test = 400

    # generate sets of pairs
    percent_to_train = 0.50
    percent_to_val = 0.25
    percent_to_test = 0.25

    pairs_count = len(all_pairs)
    video_pairs_train = int(pairs_count * percent_to_train)
    video_pairs_val = int(pairs_count * percent_to_val)
    video_pairs_test = int(pairs_count * percent_to_test)

    train_pairs = all_pairs[:video_pairs_train]
    val_pairs = all_pairs[video_pairs_train:(video_pairs_train + video_pairs_val)]
    test_pairs = all_pairs[(video_pairs_train + video_pairs_val):]

    # chose random image and save it to folder one by one
    print(f"Generation started, see results in folder {samples_folder}")

    for i in range(train):
        pair = choice(train_pairs)
        chosen_video = None
        if random.uniform(0.0, 1.0) < 0.5:
            chosen_video = pair.video_a
        else:
            chosen_video = pair.video_b

        chosen_frame_path = choice(chosen_video.generate_paths_to_frames())
        copyfile(chosen_frame_path, os.path.join(samples_folder_train, f"frame{'{:03d}'.format(i)}_{chosen_video.camera}.jpg"))
    
    for i in range(val):
        pair = choice(val_pairs)
        chosen_video = None
        if random.uniform(0.0, 1.0) < 0.5:
            chosen_video = pair.video_a
        else:
            chosen_video = pair.video_b

        chosen_frame_path = choice(chosen_video.generate_paths_to_frames())
        copyfile(chosen_frame_path, os.path.join(samples_folder_val, f"frame{'{:03d}'.format(i)}_{chosen_video.camera}.jpg"))

    for i in range(test):
        pair = choice(test_pairs)
        chosen_video = None
        if random.uniform(0.0, 1.0) < 0.5:
            chosen_video = pair.video_a
        else:
            chosen_video = pair.video_b

        chosen_frame_path = choice(chosen_video.generate_paths_to_frames())
        copyfile(chosen_frame_path, os.path.join(samples_folder_test, f"frame{'{:03d}'.format(i)}_{chosen_video.camera}.jpg"))

    print(f"Samples successfully genenerated")
    print("Script terminating")
    
