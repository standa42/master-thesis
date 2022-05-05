import gc
import datetime

from config.Config import Config
from src.helpers.helper_functions import *
from src.model.yolo_model import YoloModel

from src.data.video.video_dataset import Video_dataset

# Description:
#
# Extracts crops of objects from frames in Frame folder and puts them into Crops folder
# TODO: as of now, this script processes all videos at once
#       => should be refactored to continue previous work, or at least let user select what/how much they want to process
#       inspiration can be found in scale_down_crops.py
# TODO: let user select which object labels should be processed/ommited
#       as of now, Wheel label is hardcoded as the only one accepted

print("Script started")

# load dataset and model
print("Loading object detection model")
dataset = Video_dataset()
yolo_model = YoloModel('trackingv2')

safe_mkdir(Config.DataPaths.CropsFolder)

all_videos = dataset.get_all_videos() 

print("Processing started")
print(f"Total number of videos is {len(all_videos)}")

# go through every video, frame, recognized bounding box
for video_idx, video in enumerate(all_videos):
    frames = video.generate_frames_from_folder()
    print(f"Processing video started: {video_idx}, {video.datetime_camera}, current time: {datetime.datetime.now()}")
    
    for frame_idx, frame in enumerate(frames):
        # get detected objects from model
        for bounding_box_index, bounding_box in enumerate(yolo_model.get_bounding_boxes(frame)):
            # currently we are interested only in detection of the wheels
            if bounding_box.classification != "Wheel":
                continue
            # center, get crop of size (770,770) and save it to Crop folder
            bounding_box.make_centered_wheel_bounding_box()
            cropped_image = bounding_box.get_crop_from_image(frame)
            cropped_image.save(Config.DataPaths.CropsFolder + f"{video.datetime_camera}_frame{'{:03d}'.format(frame_idx)}_bb{bounding_box_index}_Wheel.png")
    
    print(f"Processing video ended: {video_idx}, {video.datetime_camera}, current time: {datetime.datetime.now()}")
    # ensure cleanup after every processed video
    frames.clear()
    gc.collect()

print("Processing ended") 
print("Script ended")