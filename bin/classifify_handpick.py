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

if __name__ == "__main__":

    # Declare an augmentation pipeline
    transform = A.Compose([
        A.RandomCrop(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.1),
        # A.Blur(blur_limit=3, always_apply=False, p=0.1),
        A.CoarseDropout(max_holes=3, min_holes=1, max_height=20, max_width=20, fill_value=0, always_apply=False, p=0.30),
        A.HueSaturationValue(hue_shift_limit=180, sat_shift_limit=5, val_shift_limit=5, always_apply=False, p=0.5),
        A.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, always_apply=False, p=0.2),
        A.Rotate(limit=2,always_apply=False, p=0.5),
        A.MotionBlur(p=0.05),
        A.OpticalDistortion(p=0.05)
    ])

    # open dataset folder
    dataset_folder = Config.DataPaths.UniqueRimsCollageDataset
    safe_mkdir(dataset_folder)

    # list all dirs
    types_folders = []
    for folder in os.listdir(dataset_folder):
        types_folders.append(folder)

    types_folders.remove('scooter')

    # mapping
    labels_mapping = types_folders[:]

    # prepare data holders
    train_ds = []
    train_labels = []
    val_ds = []
    val_labels = []

    # 180 30 15
    # 175 75 15
    train_ds_size = 10
    train_ds_augmented = 10
    val_ds_size = 10 

    safe_mkdir_clean(Config.DataPaths.WheelClassificationAugmentation)
    saving_counter = 0
    # load data
    for folder in types_folders:
        print(f"Processing folder {folder}")
        files = os.listdir(dataset_folder + folder + '/')
        random.shuffle(files)
        val_selection = files[:val_ds_size]
        train_selection = files[val_ds_size:(val_ds_size+train_ds_size-train_ds_augmented)]

        safe_mkdir_clean(Config.DataPaths.WheelClassificationAugmentation + folder + "/")

        label = folder

        for f in train_selection:
            img = cv2.imread(dataset_folder + folder + '/' + f)
            
            
            saving_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            saving_image = Image.fromarray(saving_image)
            saving_image.save(Config.DataPaths.WheelClassificationAugmentation + label + "/" + f"train_{label}_{saving_counter}.png")
            saving_counter = saving_counter + 1

            img = img/255.0
            train_ds.append(img)
            train_labels.append(label)

            
        for i in range(max(train_ds_size - len(train_selection), train_ds_augmented)):
            random_f = random.choice(files)
            random_img = cv2.imread(dataset_folder + folder + '/' + random_f)
            transformed = transform(image=random_img)
            transformed_image = transformed["image"]

            saving_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
            saving_image = Image.fromarray(saving_image)
            saving_image.save(Config.DataPaths.WheelClassificationAugmentation + label + "/" + f"augmentation_{label}_{saving_counter}.png")
            saving_counter = saving_counter + 1

            train_ds.append(transformed_image/255.0)
            train_labels.append(label)

        for f in val_selection:
            img = cv2.imread(dataset_folder + folder + '/' + f)
            label = folder

            saving_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            saving_image = Image.fromarray(saving_image)
            saving_image.save(Config.DataPaths.WheelClassificationAugmentation + label + "/" + f"validation_{label}_{saving_counter}.png")
            saving_counter = saving_counter + 1

            img = img/255.0
            val_ds.append(img)
            val_labels.append(label)

            
    
    # remap labels 
    for i in range(len(train_labels)):
         train_labels[i] = labels_mapping.index(train_labels[i])

    for i in range(len(val_labels)):
         val_labels[i] = labels_mapping.index(val_labels[i])

    # reshuffle dataset to mix types in the list 
    print("Shuffling dataset")
    zip_train = list(zip(train_ds, train_labels))
    random.shuffle(zip_train)
    train_ds, train_labels = zip(*zip_train)

    # convert to numpy
    train_ds = np.array(train_ds)
    train_labels = np.array(train_labels)

    val_ds = np.array(val_ds)
    val_labels = np.array(val_labels)

    np.reshape(train_ds, (-1,256,256,3))
    np.reshape(val_ds, (-1,256,256,3))

    # classes count
    num_classes = len(labels_mapping)

    print(f"num_classes: {num_classes}, and labels_mapping is {labels_mapping}")

    # resnet config
    print("Configuring resnet")
    from tensorflow.keras.applications import ResNet50, VGG16

    model = tf.keras.Sequential()
    img_shape = (256, 256, 3)
    base_model = ResNet50(include_top=False, input_shape=img_shape, weights = 'imagenet')

    how_many_layers_to_train = 40
    for layer in base_model.layers[:-how_many_layers_to_train]: #175
        layer.trainable = False

    for layer in base_model.layers[-how_many_layers_to_train:]: #175
        layer.trainable = True

    base_model.summary()

    model.add(base_model)

    # additional layers
    # model.add(tf.keras.layers.Conv2D(2048, 3, padding='same', activation='relu'))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.Conv2D(2048, 3, padding='same', activation='relu'))
    # model.add(tf.keras.layers.GlobalAveragePooling2D())
    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(256, activation='relu'))
    # model.add(tf.keras.layers.Dropout(.4))
    # model.add(tf.keras.layers.Dense(128, activation='relu'))
    # model.add(tf.keras.layers.Dropout(.2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                    optimizer='adam', 
                    metrics=['accuracy'])

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="./model/rims_classification_checkpoints2/checkpoint2",
                                                    # save_weights_only=True,
                                                    monitor='val_accuracy',
                                                    mode='max',
                                                    save_best_only=True,
                                                    verbose=1)

    model.load_weights("./model/rims_classification_checkpoints/checkpoint")
    model.save("./model/rims_classification_checkpoints2/checkpoint2")

    # prediction = model.predict(val_ds)
    # print(f"type of prediction: {type(prediction)}, values: {prediction}")
    # prediction = prediction.argmax(axis=-1)
    # print(f"type of prediction: {type(prediction)}, values: {prediction}")


    # fit
    epochs=30

    print("Training")
    history = model.fit(
    train_ds,
    train_labels,
    validation_data = (val_ds, val_labels),
    epochs=epochs,
    batch_size= 8,
    callbacks=[cp_callback]
    # TODO: class weights
    )

    print("Creating folders for inference")
    safe_mkdir_clean(Config.DataPaths.WheelClassificationFolder)
    for label in labels_mapping:
        safe_mkdir_clean(Config.DataPaths.WheelClassificationFolder + str(label) + "/")

    print("Shuffling and inferencing data")
    scaled_down_files = os.listdir(Config.DataPaths.ScaledDownCropsFolder)
    random.shuffle(scaled_down_files)
    for f in scaled_down_files:
        image_path = Config.DataPaths.ScaledDownCropsFolder + f
        image = cv2.imread(image_path)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
        image = image/255.0
        image = np.array([image])
        np.reshape(image, (-1,256,256,3))
        labels = model.predict(image)[0]
        label = np.argmax(labels)
        label = labels_mapping[label]
        copyfile(image_path, Config.DataPaths.WheelClassificationFolder + str(label) + "/" + f)



