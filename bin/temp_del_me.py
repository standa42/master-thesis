import os
import cv2
import uuid
import zipfile
from pathlib import Path
from random import choice
from shutil import copyfile

from config.Config import Config
from src.helpers.helper_functions import *

print("started")

files = [f for f in os.listdir(Config.DataPaths.CropsFolder)]

import random
from shutil import copyfile

random.shuffle(files)

for f in files[:500]:
    copyfile(Config.DataPaths.CropsFolder + f, Config.DataPaths.CropsFolder500Random + f)

print("complete")
