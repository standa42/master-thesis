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
import cv2
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

    unique_files = [Config.DataPaths.UniqueRimsCollage + f for f in listdir(Config.DataPaths.UniqueRimsCollage) if isfile(join(Config.DataPaths.UniqueRimsCollage, f))]

    # Declare an augmentation pipeline
    transform = A.Compose([
        A.RandomCrop(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Blur(blur_limit=3, always_apply=False, p=0.3),
        A.CoarseDropout(max_holes=3, min_holes=1, max_height=75, max_width=75, fill_value=0, always_apply=False, p=0.25),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=15, always_apply=False, p=0.3),
        A.RGBShift(r_shift_limit=12, g_shift_limit=12, b_shift_limit=12, always_apply=False, p=0.3),
        A.Rotate(limit=20,always_apply=False, p=0.5),
        A.MotionBlur(p=0.1),
        A.OpticalDistortion(p=0.4)
    ])

    # prepare training & validation dataset
    # validation should be our existing representatives
    # augmented rest should be training
    # convert to [0,1] interval


    # train model



    # load low-res images
    # scale them to correct dimension
    # convert to [0,1] interval

    random.shuffle(unique_files)
    # unique_files = unique_files[:10]

    train_ds = []
    train_labels = []
    val_ds = []
    val_labels = []

    # safe_mkdir_clean(Config.DataPaths.UniqueRimsAugmentation)

    for u_f in unique_files:
        img = cv2.imread(u_f)

        label = int( u_f.split('/')[-1].split('.')[0] )

        img2 = cv2.resize(img, (256,256), interpolation=cv2.INTER_CUBIC)
        val_ds.append(img2/255.0) #/255.0
        val_labels.append(label)

        img = cv2.resize(img, (290,290), interpolation=cv2.INTER_CUBIC) # was 280x280
        
        for i in range(150):
            transformed = transform(image=img)
            transformed_image = transformed["image"]
            

            train_ds.append(transformed_image/255.0) #/255.0
            train_labels.append(label)

            # transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
            # transformed_image = Image.fromarray(transformed_image)
            # transformed_image.save(Config.DataPaths.UniqueRimsAugmentation + f"{u_f.split('/')[-1].split('.')[0]}___{i}.png")

    label_mapping = val_labels[:]

    for i in range(len(train_labels)):
         train_labels[i] = label_mapping.index(train_labels[i])

    for i in range(len(val_labels)):
         val_labels[i] = label_mapping.index(val_labels[i])

    zip_train = list(zip(train_ds, train_labels))
    random.shuffle(zip_train)
    train_ds, train_labels = zip(*zip_train)

    train_ds = np.array(train_ds)
    train_labels = np.array(train_labels)
    # train_labels = to_categorical(train_labels)

    val_ds = np.array(val_ds)
    val_labels = np.array(val_labels)
    # val_labels = to_categorical(val_labels)

    np.reshape(train_ds, (-1,256,256,3))
    np.reshape(val_ds, (-1,256,256,3))

    num_classes = len(val_labels)

    # model = Sequential([
    # layers.Conv2D(16, 3, padding='same', activation='relu',input_shape=(256, 256, 3)),
    # layers.MaxPooling2D(),
    # layers.Conv2D(32, 3, padding='same', activation='relu'),
    # layers.MaxPooling2D(),
    # layers.Conv2D(64, 3, padding='same', activation='relu'),
    # layers.MaxPooling2D(),
    # layers.Conv2D(128, 3, padding='same', activation='relu'),
    # layers.GlobalAveragePooling2D(),
    # layers.Dense(num_classes, activation='softmax')
    # ])

    # model.compile(optimizer='adam',
    #             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #             metrics=['accuracy'])

    from tensorflow.keras.applications import ResNet50, VGG16

    model = tf.keras.Sequential()
    img_shape = (256, 256, 3)

    base_model = ResNet50(include_top=False, input_shape=img_shape, weights = 'imagenet')
    for layer in base_model.layers:
        layer.trainable = False

    base_model.summary()

    model.add(base_model)

    # model.add(tf.keras.layers.Reshape(target_shape=(64,64,18)))
    # model.add(tf.keras.layers.Conv2D(64,kernel_size=(3,3),name='Conv2d'))

    model.add(tf.keras.layers.Conv2D(2048, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                    optimizer='adam', 
                    metrics=['accuracy'])


    epochs=10

    history = model.fit(
    train_ds,
    train_labels,
    validation_data = (val_ds, val_labels),
    epochs=epochs,
    batch_size= 8
    )



    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    safe_mkdir_clean(Config.DataPaths.WheelClassificationFolder)
    for label in label_mapping:
        safe_mkdir_clean(Config.DataPaths.WheelClassificationFolder + str(label) + "/")

    for f in os.listdir(Config.DataPaths.ScaledDownCropsFolder):
        image_path = Config.DataPaths.ScaledDownCropsFolder + f
        image = cv2.imread(image_path)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
        image = image/255.0
        image = np.array([image])
        np.reshape(image, (-1,256,256,3))
        labels = model.predict(image)[0]
        label = np.argmax(labels)
        label = label_mapping[label]
        copyfile(image_path, Config.DataPaths.WheelClassificationFolder + str(label) + "/" + f)















































# import gc
# import datetime
# import random
# import cv2

# from PIL import Image
# from os import listdir
# from os.path import isfile, join

# from config.Config import Config
# from src.helpers.helper_functions import *
# from src.model.yolo_model import YoloModel

# from src.data.video.video import Video
# from src.data.video.video_pair import Video_pair
# from src.data.video.video_dataset import Video_dataset

# import matplotlib.pyplot as plt
# import cv2
# from skimage.feature import hog
# from skimage import data, exposure

# from sklearn.mixture import GaussianMixture
# from sklearn.cluster import KMeans
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# from skimage.feature import (match_descriptors, corner_harris,
#                              corner_peaks, ORB, plot_matches)

# from shutil import copyfile
# import albumentations as A

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential

# import numpy as np

# if __name__ == "__main__":

#     safe_mkdir_clean(Config.DataPaths.UniqueRimsAugmentation)
#     unique_files = [Config.DataPaths.UniqueRimsCollage + f for f in listdir(Config.DataPaths.UniqueRimsCollage) if isfile(join(Config.DataPaths.UniqueRimsCollage, f))]

#     # Declare an augmentation pipeline
#     transform = A.Compose([
#         A.RandomCrop(width=256, height=256),
#         A.HorizontalFlip(p=0.5),
#         A.RandomBrightnessContrast(p=0.2),
#         A.Blur(blur_limit=3, always_apply=False, p=0.2),
#         A.Cutout(num_holes=8, max_h_size=20, max_w_size=20, fill_value=0, always_apply=False, p=0.3),
#         A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, always_apply=False, p=0.5),
#         A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, always_apply=False, p=0.5)
#     ])

#     train_ds = []

#     for u_f in unique_files:
#         img = cv2.imread(u_f)
#         img = cv2.resize(img, (280,280), interpolation=cv2.INTER_CUBIC)
        
#         img2 = cv2.resize(img, (256,256), interpolation=cv2.INTER_CUBIC)
#         train_ds.append(img2/255.0)

#         for i in range(10):
#             transformed = transform(image=img)
#             transformed_image = transformed["image"]
#             transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
#             transformed_image = Image.fromarray(transformed_image)

#             transformed_image.save(Config.DataPaths.UniqueRimsAugmentation + f"{i}_{u_f.split('/')[-1]}")

#     val_ds = train_ds


#     num_classes = len(unique_files)

#     model = Sequential([
#     layers.Conv2D(16, 3, padding='same', activation='relu',input_shape=(256, 256, 3)),
#     layers.MaxPooling2D(),
#     layers.Conv2D(32, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Conv2D(64, 3, padding='same', activation='relu'),
#     layers.MaxPooling2D(),
#     layers.Flatten(),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(num_classes)
#     ])

#     model.compile(optimizer='adam',
#                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                 metrics=['accuracy'])

#     epochs=10

#     train_ds = np.array(train_ds)
#     np.reshape(train_ds, (25,256,256,3))

#     history = model.fit(
#     train_ds,
#     np.array( list(range(25)) ),
#     validation_data = (train_ds, np.array( list(range(25)) )),

#     # validation_data=val_ds,
#     epochs=epochs
#     )

#     acc = history.history['accuracy']
#     val_acc = history.history['val_accuracy']

#     loss = history.history['loss']
#     val_loss = history.history['val_loss']

#     epochs_range = range(epochs)

#     plt.figure(figsize=(8, 8))
#     plt.subplot(1, 2, 1)
#     plt.plot(epochs_range, acc, label='Training Accuracy')
#     plt.plot(epochs_range, val_acc, label='Validation Accuracy')
#     plt.legend(loc='lower right')
#     plt.title('Training and Validation Accuracy')

#     plt.subplot(1, 2, 2)
#     plt.plot(epochs_range, loss, label='Training Loss')
#     plt.plot(epochs_range, val_loss, label='Validation Loss')
#     plt.legend(loc='upper right')
#     plt.title('Training and Validation Loss')
#     plt.show()
