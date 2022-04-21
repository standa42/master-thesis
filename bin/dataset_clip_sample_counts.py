from PIL import Image, ImageDraw, ImageFont
from os import listdir
from os.path import isfile, join
from shutil import copyfile
import random

from src.helpers.helper_functions import *
from config.Config import Config

clip_threshold = 200

dataset_folder = Config.DataPaths.UniqueRimsCollageDataset
dataset_clipped_folder = Config.DataPaths.UniqueRimsCollageDatasetClipped

# ensure folders exist
safe_mkdir(dataset_folder)
safe_mkdir_clean(dataset_clipped_folder)

# go through each label of original handpicked dataset
for folder in listdir(dataset_folder):
    if folder.isdigit() or folder == 'unrecognized':
        folder_path = join(dataset_folder, folder)
        files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]
        random.shuffle(files)
        files = files[:clip_threshold]
        safe_mkdir_clean(join(dataset_clipped_folder, folder))
        for f in files:
            copyfile(join(dataset_folder, folder, f), join(dataset_clipped_folder, folder, f))

