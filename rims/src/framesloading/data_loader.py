# data structure in frames folder is date/time/A-B/allinfo_1-600.img

# get data for:
#    day -> list all days
#    day/video=1 -> one day
#    day/video=1/A -> one camera
#    day/video=1/A/1-600 -> one frame

# load all data as file names and then they can be loaded on demand

import os

class rim_dataset:
    def __init__(self, days):
        self.days = days

class day_data:
    def __init__(self, ten_min_data, day):
        self.ten_min_data = ten_min_data
        self.day = day

class ten_min_data:
    def __init__(self, a, b, time):
        self.a = a
        self.b = b
        self.cameras = [a,b]
        self.time = time

class camera_data:
    def __init__(self, frames, camera_type):
        self.frames = frames
        self.camera_type = camera_type

class frame_data:
    def __init__(self, filename, path):
        self.filename = filename
        self.path = path
        self.number = int(filename.split('.')[0].split('e')[1])
        self.image = None

class frames_loader:
    def __init__(self):
        self.video_frames_path = "./rims/data/frames/"
        self.dataset = None

    def load_dataset(self):
        # iterate through days
        days = []
        for day_folder in os.listdir(self.video_frames_path):
            day_folder_path = os.path.join(self.video_frames_path, f"{day_folder}/")
            # iterate through times
            times = []
            for time_folder in os.listdir(day_folder_path): 
                time_folder_path = os.path.join(day_folder_path, f"{time_folder}/")
                # iterate through cameras
                cameras = []
                for camera_folder in os.listdir(time_folder_path): 
                    camera_folder_path = os.path.join(time_folder_path, f"{camera_folder}/")
                    # iterate through frames
                    frames = []
                    for frame_file in os.listdir(camera_folder_path): 
                        frame_file_path = os.path.join(camera_folder_path, f"{frame_file}")
                        frame = frame_data(frame_file, frame_file_path)
                        frames.append(frame)
                    camera = camera_data(frames, camera_folder)
                    cameras.append(camera)
                time = ten_min_data(cameras[0], cameras[1], time_folder)
                times.append(time)
            day = day_data(times, day_folder)
            days.append(day)
        self.dataset = rim_dataset(days)
