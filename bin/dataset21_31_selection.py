import os
import random
from shutil import copyfile
from config.Config import Config
from src.helpers.helper_functions import *

dataset31_folder = Config.DataPaths.Dataset31
dataset31_not_selected_folder = Config.DataPaths.Dataset31NotSelected

safe_mkdir(dataset31_not_selected_folder)
safe_mkdir_clean(dataset31_folder)

labels_in_dataset_31 = os.listdir(dataset31_not_selected_folder)

train_count = 100
val_count = 25
test_count = 25

for label in labels_in_dataset_31:
    label_folder = dataset31_not_selected_folder + label + '/'
    
    label_train_folder = label_folder + 'train/'
    label_val_folder = label_folder + 'val/'
    label_test_folder = label_folder + 'test/'

    train_files = os.listdir(label_train_folder)
    val_files = os.listdir(label_val_folder)
    test_files = os.listdir(label_test_folder)

    if len(train_files) < train_count:
        sampled_train_files = train_files
    else:
        sampled_train_files = random.sample(train_files, train_count)
    if len(val_files) < val_count:
        sampled_val_files = val_files
    else:
        sampled_val_files = random.sample(val_files, val_count)
    if len(test_files) < test_count:
        sampled_test_files = test_files
    else:
        sampled_test_files = random.sample(test_files, test_count)

    target_label_folder = dataset31_folder + label + '/'
    target_label_train_folder = target_label_folder + 'train/'
    target_label_val_folder = target_label_folder + 'val/'
    target_label_test_folder = target_label_folder + 'test/'

    safe_mkdir_clean(target_label_folder)
    safe_mkdir_clean(target_label_train_folder)
    safe_mkdir_clean(target_label_val_folder)
    safe_mkdir_clean(target_label_test_folder)

    for sample in sampled_train_files:
        source_file = label_train_folder + sample
        target_file = target_label_train_folder + sample
        copyfile(source_file, target_file)

    for sample in sampled_val_files:
        source_file = label_val_folder + sample
        target_file = target_label_val_folder + sample
        copyfile(source_file, target_file)

    for sample in sampled_test_files:
        source_file = label_test_folder + sample
        target_file = target_label_test_folder + sample
        copyfile(source_file, target_file)














