import os
import cv2
import time
import datetime

from pathlib import Path
import os

import cv2
from IPython.display import clear_output, display

from src.videoparsing.video_metadata import video_metadata
from src.helpers.helper_functions import *

class video_dataset:
    def __init__(self):
        self.dataset_folder_path = "./rims/data/video/"
        self.video_frames_path = "./rims/data/frames2/"

    def standalone_cmd_interface(self):
        """ Cmd dialog used to select days to process """
        # Introductions and listing of all available video data
        print("Please select which day do you want to process to frames")
        print("By writing 'all', you select all of them")
        print("By writing concrete number like '5', or list delimited by commas like '2,3,5,8', you can select them individually")
        print()
        print("Videos for following days are available: ")
        counter = 1
        for day_folder in os.listdir(self.dataset_folder_path):
            print(str(counter) + " - " + day_folder)
            counter = counter + 1
        print("Your choice: ")

        # Choice of user via text input
        while True:
            selected_list = input()
            selected_list = selected_list.strip()
            
            if (selected_list == "all"):
                selected_list = list(range(1,counter))
                break
            else: 
                try:
                    selected_list = selected_list.split(",")
                    selected_list = map(lambda x: int(x.strip()), selected_list)
                    break
                except:
                    print("Invalid input, please try again:")
        
        # Translation of choice to concrete days
        selected_days = []
        counter = 1
        for day_folder in os.listdir(self.dataset_folder_path):
            if (counter in selected_list):
                selected_days.append(day_folder)
            counter = counter + 1

        # Processing
        self.parse_videos_into_frames(selected_days)

    def process_all_videos(self):
        """ Processes all days of video it finds, inteded to be integrated into pipeline """
        selected_days_all =  list(os.listdir(self.dataset_folder_path))
        self.parse_videos_into_frames(selected_days_all)


    def get_videos_metadata(self, selected_days):
        """ Get complete information about all video files in selected days """
        all_videos = []
        for day_folder in selected_days: # in day_folders
            for video in os.listdir(os.path.join(self.dataset_folder_path, day_folder)): # in videos
                video_name_split_period = video.split('.')
                video_name_split_underscore = video_name_split_period[0].split('_')

                year   = video_name_split_underscore[0]
                month  = video_name_split_underscore[1]
                day    = video_name_split_underscore[2]
                hour   = video_name_split_underscore[3]
                minute = video_name_split_underscore[4]
                second = video_name_split_underscore[5]
                camera = video_name_split_underscore[6]
                file_extension = video_name_split_period[1]

                single_video = video_metadata(year, month, day, hour, minute, second, camera, file_extension)
                all_videos.append(single_video)
                # print(f"Found video {year}-{month}-{day}-{hour}-{minute}-{second}-{camera}.{file_extension}")
        return all_videos

    def parse_videos_into_frames(self, selected_days):
        """ Parsing of all video files in selected days """
        print("VIDEO TO FRAMES")
        print("processing started")
        print("you will get status update for every 10th video with expected time of processing")
        print("(during development, it was about 30-60 seconds per video)")

        # get all video metadata for selected days
        all_videos_metadata = self.get_videos_metadata(selected_days)

        print(f"(means rough estimate for {len(all_videos_metadata)} videos is about {len(all_videos_metadata) * 35} seconds = {'{:.2f}'.format((len(all_videos_metadata) * 35) / 60)} minutes = {'{:.2f}'.format(((len(all_videos_metadata) * 35) / 60)/60)} hours)")
        self.print_progress(len(all_videos_metadata), 35)
        # ensure video_frames folder exists
        safe_mkdir(self.video_frames_path)
        
        video_counter = 0
        avg_video_time = -1
        start = -1
        for single_video_metadata in all_videos_metadata:
            # measure time expecation for single video processing and inform user about progress
            end = time.time()
            if (start > 0):
                time_diff_sec = end - start
                avg_video_time = time_diff_sec if avg_video_time == -1 else (avg_video_time + time_diff_sec) / 2
            start = time.time()

            video_counter = video_counter + 1
            if (video_counter % 10 == 2):
                self.print_progress(len(all_videos_metadata), avg_video_time, video_counter)
            
            # check day folder exists
            video_day_path = os.path.join(self.video_frames_path, f"{single_video_metadata.day_string()}/")
            safe_mkdir(video_day_path)
            # check time folder exists
            video_time_path = os.path.join(video_day_path, f"{single_video_metadata.time_string()}/")
            safe_mkdir(video_time_path)
            # check camera folder exists
            video_camera_path = os.path.join(video_time_path, f"{single_video_metadata.camera}/")
            safe_mkdir_clean(video_camera_path)

            frames_path = video_camera_path

            # note: expectetion is that videos are 10 minutes long with 1 frame per second => 600 frames
            files_in_frame_path = [f for f in os.listdir(frames_path) if os.path.isfile(os.path.join(frames_path, f))]
            if files_in_frame_path == []:
                # print(f"Converting {single_video_metadata} to frames")
                cap_path = os.path.join(self.dataset_folder_path, f"{single_video_metadata.day_string()}/", f"{single_video_metadata}" )
                cap = cv2.VideoCapture(cap_path)
                # print(f"Cap_path is {cap_path}")
                count = 0
                # frame_string = f"frame{count}.jpg"
                # print(f"Writing to {os.path.join(frames_path, frame_string)}")
                while cap.isOpened():
                    #clear_output(wait=True)
                    # print(f"Frame {count}")
                    ret,frame = cap.read()
                    # cv2.imshow('window-name', frame)
                    cv2.imwrite(os.path.join(frames_path, f"frame{'{:03d}'.format(count)}.jpg"), frame)
                    count = count + 1
                    if count > 599:
                        break
                cap.release()
                # cv2.destroyAllWindows() # destroy all opened windows
            else:
                print(f"Video {single_video_metadata} was already converted")

    def print_progress(self, videos_count, avg_time_per_video, videos_already_processed = 0):
        all_videos_length = videos_count
        expected_seconds_remaining = (all_videos_length - videos_already_processed) * avg_time_per_video
        expected_datetime_remaining = (datetime.datetime.now() + datetime.timedelta(seconds=expected_seconds_remaining)).strftime("%Y.%m.%d %H:%M:%S")
        avg_video_time_string = "{:.2f}".format(avg_time_per_video)
        print(f"Progress: {videos_already_processed}/{all_videos_length}, one video takes about {avg_video_time_string} seconds, estimated end: {expected_datetime_remaining}")


if __name__ == "__main__":
    dataset = video_dataset()
    dataset.standalone_cmd_interface()