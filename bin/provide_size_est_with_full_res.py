import os
import random
from shutil import copyfile
from config.Config import Config
from src.helpers.helper_functions import *

crops_folder = Config.DataPaths.CropsFolder

size_estimation_train_folder = Config.DataPaths.SizeEstimationDatasetTrain
size_estimation_test_folder = Config.DataPaths.SizeEstimationDatasetTest
size_estimation_val_folder = Config.DataPaths.SizeEstimationDatasetVal

train_files = os.listdir(size_estimation_train_folder)
train_files = list(filter(lambda x: x[-4:] == ".png", train_files))

for f in train_files:
    source = crops_folder + f 
    target = size_estimation_train_folder + f 
    copyfile(source, target)



val_files = os.listdir(size_estimation_val_folder)
val_files = list(filter(lambda x: x[-4:] == ".png", val_files))

for f in val_files:
    source = crops_folder + f 
    target = size_estimation_val_folder + f 
    copyfile(source, target)



test_files = os.listdir(size_estimation_test_folder)
test_files = list(filter(lambda x: x[-4:] == ".png", test_files))

for f in test_files:
    source = crops_folder + f 
    target = size_estimation_test_folder + f 
    copyfile(source, target)







