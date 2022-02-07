import gc
import datetime

from config.Config import Config
from src.helpers.helper_functions import *
from src.model.yolo_model import YoloModel

from src.data.video.video import Video
from src.data.video.video_pair import Video_pair
from src.data.video.video_dataset import Video_dataset

if __name__ == "__main__":
    dataset = Video_dataset()
    yolo_model = YoloModel('tracking')

    safe_mkdir(Config.DataPaths.CropsFolder)

    all_video_pairs = dataset.get_all_videos()
    all_video_pairs = all_video_pairs[(16+102+20+32+18+28+2+64+92+42):] # 16 102 20 32 18 28 2 64 92 42 complete (y)

    print("Processing started")

    video_counter = 0
    for video in all_video_pairs:
        frame_counter = 0
        frames = video.generate_frames_from_folder()
        print(f"Processing video started: {video_counter + 1}, {video.datetime_camera}, current time: {datetime.datetime.now()}")
        for frame in frames:
            bounding_box_counter = 0 
            for bounding_box in yolo_model.get_bounding_boxes(frame):
                if bounding_box.classification != "pneu": # change later to Wheel (I have now trained yolo network with wrong class names)
                    continue
                bounding_box.make_centered_wheel_bounding_box()
                cropped_image = bounding_box.get_crop_from_image(frame)
                cropped_image.save(Config.DataPaths.CropsFolder + f"{video.datetime_camera}_frame{'{:03d}'.format(frame_counter)}_bb{bounding_box_counter}_Wheel.png") # bb{bounding_box_counter}_{bounding_box.classification}.png')
                bounding_box_counter = bounding_box_counter + 1
            frame_counter = frame_counter + 1
        video_counter = video_counter + 1
        print(f"Processing video ended: {video_counter}, {video.datetime_camera}, current time: {datetime.datetime.now()}")
        del frames
        gc.collect()
 
    print("Processing ended") 

    