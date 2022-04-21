import random
import cv2

from PIL import Image
from os import listdir
from os.path import isfile, join

from config.Config import Config
from src.helpers.helper_functions import *

# Description
#
# Scales down crops from Crops folder to size (256, 256) and puts them into Downscaled Crops folder

print("Script started")

safe_mkdir(Config.DataPaths.CropsFolder)
safe_mkdir(Config.DataPaths.ScaledDownCropsFolder)

# get crop files
crop_files = [f for f in listdir(Config.DataPaths.CropsFolder) if isfile(join(Config.DataPaths.CropsFolder, f))]
# get scaled down crops
scaled_down_crop_files = [f for f in listdir(Config.DataPaths.ScaledDownCropsFolder) if isfile(join(Config.DataPaths.ScaledDownCropsFolder, f))]

print(f"There is: {len(crop_files)}")
print(f"And already scaled down is: {len(scaled_down_crop_files)}")
print(f"This means that remains {len(crop_files) - len(scaled_down_crop_files)} crops to be scaled down")

# crop files that werent scaled down yet
crop_files = list(set(crop_files).difference(set(scaled_down_crop_files)))

random.shuffle(crop_files)

print("Processing started")

# downscaling and saving
counter = 0
for crop_file in crop_files:
    crop_file_path = Config.DataPaths.CropsFolder + crop_file
    crop_image = cv2.imread(crop_file_path) 
    crop_image = cv2.resize(crop_image, (256,256), interpolation=cv2.INTER_CUBIC)
    crop_image = cv2.cvtColor(crop_image, cv2.COLOR_BGR2RGB)
    crop_image = Image.fromarray(crop_image)
    crop_image.save(Config.DataPaths.ScaledDownCropsFolder + crop_file)

    counter = counter + 1
    if counter % 1000 == 500:
        print(f"Processed {counter} of images")

print("Processing ended")
print("Script ended")