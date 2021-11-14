import os
import random

from config.Config import Config
from src.data.video.video import Video
from src.data.video.video_pair import Video_pair

class Video_dataset:
    def __init__(self):
        pass

    def get_all_videos(self):
        """Returns all available videos"""
        available_days = self.get_available_days()
        video_files_paths = []

        for day in available_days:
            # get folder for certain day
            day_folder = os.path.join(Config.DataPaths.VideoFolder, day)
            # get paths to all videos in it
            video_files_paths.extend( [day_folder + '/' + f for f in os.listdir(day_folder) if os.path.isfile(os.path.join(day_folder, f))] ) 

        # create instances of Videos from the paths
        videos = [Video(video_file_path.split('/')[-1], video_file_path) for video_file_path in video_files_paths]
        return videos

    def get_all_video_pairs(self):
        """Returns all available video pairs"""
        # get all videos
        videos = self.get_all_videos()
        pairs = []

        for video in videos:
            # if video is from camera A, pair it with a video from camera B
            if video.camera == 'A':
                video_a = video
                video_b = next(v for v in videos if v.datetime == video_a.datetime and v.camera == 'B')
                pairs.append(Video_pair(video_a, video_b))

        return pairs

    def get_available_days(self):
        """Returns days that are available in dataset"""
        # go to video folder
        video_folder = Config.DataPaths.VideoFolder
        # get all days folders
        day_folders = [folder for folder in os.listdir(video_folder)]
        return day_folders

    def get_random_video_pair(self):
        """Returns random video pair from the whole dataset"""
        return random.choice(self.get_all_video_pairs())
