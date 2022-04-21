import gc
import os
import datetime
import random
import cv2

from PIL import Image
from os import listdir
from os.path import isfile, join

from config.Config import Config
from src.helpers.helper_functions import *
from src.model.yolo_model import YoloModel

from src.data.video.video import Video
from src.data.video.video_pair import Video_pair
from src.data.video.video_dataset import Video_dataset

import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure

from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from skimage.feature import (match_descriptors, corner_harris,
                             corner_peaks, ORB, plot_matches)

from shutil import copyfile
import albumentations as A

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical

import numpy as np
import copy

if __name__ == "__main__":

    images_per_class_and_transform_generated = 100

    # low augmentation - the whole pipeline gives reasonable output
    transformations = {
        "HorizontalFlip": A.HorizontalFlip(always_apply=True),
        "RandomBrightnessContrast": A.RandomBrightnessContrast(always_apply=True, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.1, 0.1)),
        "CoarseDropout": A.CoarseDropout(max_holes=10, min_holes=2, max_height=20, max_width=20, fill_value=0, always_apply=True),
        "HueSaturationValue": A.HueSaturationValue(hue_shift_limit=180, sat_shift_limit=5, val_shift_limit=5, always_apply=True),
        "RGBShift": A.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, always_apply=True),
        "Rotate": A.Rotate(limit=5, always_apply=True),
        "MotionBlur": A.MotionBlur(always_apply=True, blur_limit=(3, 7)),
        "OpticalDistortion": A.OpticalDistortion(always_apply=True, distort_limit=(-0.2, 0.2), shift_limit=(-0.05, 0.05)), 
    }

    # high augmentation - the whole pipeline gives overaugmented output
    # transformations = {
    #     "HorizontalFlip": A.HorizontalFlip(always_apply=True),
    #     "RandomBrightnessContrast": A.RandomBrightnessContrast(always_apply=True, brightness_limit=(-0.4, 0.4), contrast_limit=(-0.2, 0.2)),
    #     "CoarseDropout": A.CoarseDropout(max_holes=10, min_holes=2, max_height=20, max_width=20, fill_value=0, always_apply=True),
    #     "HueSaturationValue": A.HueSaturationValue(hue_shift_limit=180, sat_shift_limit=10, val_shift_limit=10, always_apply=True),
    #     "RGBShift": A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, always_apply=True),
    #     "Rotate": A.Rotate(limit=15, always_apply=True),
    #     "MotionBlur": A.MotionBlur(always_apply=True, blur_limit=(3, 15)),
    #     "OpticalDistortion": A.OpticalDistortion(always_apply=True, distort_limit=(-0.5, 0.5), shift_limit=(-0.1, 0.1)), 
    # }

    transformations["Combination"] = A.Compose(transformations.values())

    # load classes
    dataset_folder = Config.DataPaths.UniqueRimsCollageDataset
    dataset_classes = os.listdir(dataset_folder)
    dataset_classes.remove('scooter')
    dataset_classes = ["90"]

    # create folder to save them
    augmentation_folder = Config.DataPaths.AugmentationOfClasses
    safe_mkdir_clean(augmentation_folder)

    for dataset_class in dataset_classes:
        # create folder for class
        class_augmentation_folder = os.path.join(augmentation_folder, dataset_class + '/')
        safe_mkdir_clean(class_augmentation_folder)
        # get representative file for class
        representation_file_path = os.path.join(Config.DataPaths.UniqueRimsCollage, dataset_class + ".png")
        representation_file = cv2.imread(representation_file_path)

        # iterate transformation
        for transformation, transformation_function in transformations.items():
            class_transformation_folder = os.path.join(class_augmentation_folder, transformation + '/')
            safe_mkdir_clean(class_transformation_folder)
            # generate certain amount of files
            for i in range(images_per_class_and_transform_generated):
                file_path_for_save = os.path.join(class_transformation_folder, str(i).zfill(4) + ".png")

                transformed_obj = transformation_function(image=representation_file)
                transformed_image = transformed_obj["image"]

                saving_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
                saving_image = Image.fromarray(saving_image)
                saving_image.save(file_path_for_save)