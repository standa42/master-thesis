import os
import random
from shutil import copyfile
from config.Config import Config
from src.helpers.helper_functions import *

dataset31_folder = Config.DataPaths.Dataset31
target_folder = Config.DataPaths.WheelAndBoltsDatasetSamples
target_folder_train = target_folder + 'train/'
target_folder_val = target_folder + 'val/'
target_folder_test = target_folder + 'test/'

safe_mkdir_clean(target_folder)
safe_mkdir(target_folder_train)
safe_mkdir(target_folder_val)
safe_mkdir(target_folder_test)

labels_in_dataset_31 = os.listdir(dataset31_folder)

train_count = 500
val_count = 100
test_count = 100

train_data = []
val_data = []
test_data = []

for label in labels_in_dataset_31:
    label_folder = dataset31_folder + label + '/'
    
    label_train_folder = label_folder + 'train/'
    label_val_folder = label_folder + 'val/'
    label_test_folder = label_folder + 'test/'

    train_files = os.listdir(label_train_folder)
    val_files = os.listdir(label_val_folder)
    test_files = os.listdir(label_test_folder)

    train_files = list(map(lambda x: label_train_folder + x, train_files))
    val_files = list(map(lambda x: label_val_folder + x, val_files))
    test_files = list(map(lambda x: label_test_folder + x, test_files))

    train_data.extend(train_files)
    val_data.extend(val_files)
    test_data.extend(test_files)


train_data = random.sample(train_data, train_count)
val_data = random.sample(val_data, val_count)
test_data = random.sample(test_data, test_count)

for sample in train_data:
    source = sample
    target = target_folder_train + sample.split('/')[-1]
    copyfile(source, target)

for sample in val_data:
    source = sample
    target = target_folder_val + sample.split('/')[-1]
    copyfile(source, target)

for sample in test_data:
    source = sample
    target = target_folder_test + sample.split('/')[-1]
    copyfile(source, target)















