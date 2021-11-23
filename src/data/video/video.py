import os
from pathlib import Path
import cv2

from config.Config import Config
from src.helpers.helper_functions import *

class Video:
    def __init__(self, filename, path):
        filename_without_extension = filename.replace(".mp4", "")
        filename_split = filename_without_extension.split('_')
        
        self.date = filename_split[0] + "_" + filename_split[1] + "_" + filename_split[2]
        self.time = filename_split[3] + "_" + filename_split[4] + "_" + filename_split[5]
        self.camera = filename_split[6]
        self.datetime = self.date + "_" + self.time
        self.datetime_camera = self.date + "_" + self.time + "_" + self.camera
        self.path = path

    def is_parsed(self):
        """Checks whether video is parsed to the frames folder on the disc"""
        camera_folder_path = self._get_frames_path_camera_folder()
        # check that folder for that video exists
        if(Path(camera_folder_path).exists()):
            # if it contains 600 files, which it should as it is 600 frames per video - consider it parsed ok
            if len(os.listdir(camera_folder_path)) == 600:
                return True
        return False 

    def parse(self):
        """Parses video to the frames folder on the disc"""
        # return if already parsed
        if self.is_parsed():
            return

        # ensure all necessary folders on the way exist
        safe_mkdir(self._get_frames_path_folder())
        safe_mkdir(self._get_frames_path_day_folder())
        safe_mkdir(self._get_frames_path_time_folder())
        camera_folder = self._get_frames_path_camera_folder()
        safe_mkdir_clean(camera_folder)
        
        # parse and save frames
        cap = cv2.VideoCapture(self.path)
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            cv2.imwrite(camera_folder + f"frame{'{:03d}'.format(count)}.jpg", frame)
            count = count + 1
            if count >= 600:
                break
        cap.release()

    def generate_paths_to_frames(self):
        """Generates paths to all frames in the video, sorted by time ascending"""
        # takes video folder where
        return [self._get_frames_path_camera_folder() + f"frame{'{:03d}'.format(count)}.jpg" for count in range(600)]

    def generate_frames_from_video(self):
        """Returns a generator of frames directly from the video"""
        cap = cv2.VideoCapture(self.path)
        count = 0
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            frames.append(frame)
            count = count + 1
            if count >= 600:
                break
            cap.release()
        return frames

    def generate_frames_from_folder(self):
        """Returns a generator of frames from the folder (already parsed on disc)"""
        frames_paths = self.generate_paths_to_frames()
        frames = []
        for frame_path in frames_paths:
            frames.append(cv2.imread(frame_path))
        return frames

    # just helpers with paths
    def _get_frames_path_folder(self):
        return Config.DataPaths.FramesFolder

    def _get_frames_path_day_folder(self):
        return Config.DataPaths.FramesFolder + self.date + '/'

    def _get_frames_path_time_folder(self):
        return Config.DataPaths.FramesFolder + self.date + '/' + self.time + '/'

    def _get_frames_path_camera_folder(self):
        return Config.DataPaths.FramesFolder + self.date + '/' + self.time + '/' + self.camera + '/'