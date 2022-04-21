from PIL import Image, ImageDraw, ImageFont
from os import listdir
from os.path import isfile, join
from shutil import copyfile
import random

from src.helpers.helper_functions import *
from config.Config import Config

clip_threshold = 200

dataset_clipped_folder = Config.DataPaths.UniqueRimsCollageDatasetClipped
dataset_clipped_geometry_folder = Config.DataPaths.UniqueRimsCollageDatasetClippedGeometryOnly

# ensure folders exist
safe_mkdir(dataset_clipped_folder)
safe_mkdir(dataset_clipped_geometry_folder)

# go through each label of original handpicked dataset
folders = listdir(dataset_clipped_folder)
for folder in folders:
    if folder.isdigit() and int(folder) % 10 == 1:
        continue

    if folder.isdigit() or folder == 'unrecognized':
        folder_path = join(dataset_clipped_folder, folder)
        files = [join(dataset_clipped_folder, folder + '/', f) for f in listdir(folder_path) if isfile(join(folder_path, f))]
        
        if folder.isdigit():
            folder_plus_one = str(int(folder)+1)
            if folder_plus_one in folders:
                folder_plus_one_path = join(dataset_clipped_folder, folder_plus_one)
                files.extend([join(dataset_clipped_folder, folder_plus_one + '/', f) for f in listdir(folder_plus_one_path) if isfile(join(folder_plus_one_path, f))])
                random.shuffle(files)

        files = files[:clip_threshold]
        safe_mkdir_clean(join(dataset_clipped_geometry_folder, folder))
        for f in files:
            copyfile(f, join(dataset_clipped_geometry_folder, folder, f.split('/')[-1]))

