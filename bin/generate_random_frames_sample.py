import os
import cv2
import uuid
import zipfile
from pathlib import Path
from random import choice
from shutil import copyfile

from config.Config import Config
from src.helpers.helper_functions import *

from src.data.video.video import Video
from src.data.video.video_pair import Video_pair
from src.data.video.video_dataset import Video_dataset

if __name__ == "__main__":
    # get all videos
    dataset = Video_dataset()
    all_videos = dataset.get_all_videos()

    # select number of frames to generate
    print("Please write sample size in number of frames: ")
    selection = input()
    selection = selection.strip()     
    print(f"Your choice was: {selection}")

    selection_int = None

    try:
        selection_int = int(selection)
        
    except:
        print('Your choice is either not integer, or it is larger than available days')
        print('Terminating script')
        quit()

    # ensure directiories are made
    safe_mkdir(Config.DataPaths.RandomFrameSampleFolder)
    folder_for_dataset = Config.DataPaths.RandomFrameSampleFolder + f"sample_of_{selection_int}_{uuid.uuid4().hex[:6]}"
    safe_mkdir_clean(folder_for_dataset)

    print(f"Generation started, target location is folder: {folder_for_dataset}")

    # chose random image and save it to folder one by one
    for i in range(selection_int):
        chosen_video = choice(all_videos)
        chosen_frame_path = choice(chosen_video.generate_paths_to_frames())
        copyfile(chosen_frame_path, os.path.join(folder_for_dataset, f"frame{'{:03d}'.format(i)}_{chosen_video.camera}.jpg"))
        # print(f"Video {chosen_video.datetime}, frame {chosen_frame_path.split('/')[-1]}")

    print(f"Sample genenerated")
    print(f"Press any key to continue..")
    input()
    
