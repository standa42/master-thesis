import gc
import os
import datetime
import random
import cv2
import sys
import time

from PIL import Image
from os import listdir
from os.path import isfile, join

import sklearn
from torch import _test_serialization_subcmul

from config.Config import Config
from src.helpers.helper_functions import *
from src.model.yolo_model import YoloModel

from src.data.video.video import Video
from src.data.video.video_pair import Video_pair
from src.data.video.video_dataset import Video_dataset

import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure

import sklearn
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from skimage.feature import (match_descriptors, corner_harris,
                             corner_peaks, ORB, plot_matches)
from sklearn.svm import LinearSVC, SVC                           
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib as mpl

from shutil import copyfile
import albumentations as A

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.applications import ResNet50, VGG16
import tensorflow.keras.applications

import numpy as np

# sys.stdout = open("./model/rims_classification_checkpoints/output.txt", "w")


# dataset configuration
train_ds_size = 150 # 180
train_ds_augmented = 30
val_ds_size = 10
dataset_choice = 'color_geometry'
# dataset_choice = 'geometry'  color_geometry
only_labels_with_full_count = False # easy = True, hard = False

# model configuration
# ResNet50
# EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
# EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, EfficientNetV2B3, EfficientNetV2B4, EfficientNetV2B5, EfficientNetV2B6, EfficientNetV2B7
model_name = 'EfficientNetB0' # 
weights = 'imagenet' # None, 'imagenet'
layers_to_unfreeze = 0
epochs=2
batch_size = 8
saving_callback = False

class ClassificationDataset:

    def init(self, dataset_choice, only_labels_with_full_count, train_ds_size, train_ds_augmented, val_ds_size):
        print("Dataset initialization")
        print(f"parameters - dataset_choice: {dataset_choice}, only_labels_with_full_count: {only_labels_with_full_count}")
        print(f"train_ds_size: {train_ds_size}, train_ds_augmented: {train_ds_augmented}, val_ds_size: {val_ds_size}")

        self.dataset_choice = dataset_choice
        self.only_labels_with_full_count = only_labels_with_full_count
        self.train_ds_size = train_ds_size
        self.train_ds_augmented = train_ds_augmented
        self.val_ds_size = val_ds_size

        self.init_dataset_folder()
        self.init_augmentation()
        self.init_labels_mapping()
        safe_mkdir_clean(Config.DataPaths.WheelClassificationAugmentation)


    def init_dataset_folder(self):
        if self.dataset_choice == 'color_geometry':
            self.dataset_folder = Config.DataPaths.UniqueRimsCollageDatasetClipped
        elif self.dataset_choice == 'geometry':
            self.dataset_folder = Config.DataPaths.UniqueRimsCollageDatasetClippedGeometryOnly
        safe_mkdir(self.dataset_folder)

    def init_augmentation(self):
        # Declare an augmentation pipeline
        transform = {
            "RandomBrightnessContrast": A.RandomBrightnessContrast(always_apply=True, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.1, 0.1)),
            "CoarseDropout": A.CoarseDropout(max_holes=10, min_holes=2, max_height=20, max_width=20, fill_value=0, always_apply=True),
            "HueSaturationValue": A.HueSaturationValue(hue_shift_limit=180, sat_shift_limit=5, val_shift_limit=5, always_apply=True),
            "RGBShift": A.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, always_apply=True),
            "Rotate": A.Rotate(limit=5, always_apply=True),
            "MotionBlur": A.MotionBlur(always_apply=True, blur_limit=(3, 7)),
            "OpticalDistortion": A.OpticalDistortion(always_apply=True, distort_limit=(-0.2, 0.2), shift_limit=(-0.05, 0.05)), 
        }
        transform = A.Compose(transform.values())   
        self.transform = transform

    def init_labels_mapping(self):
        # open dataset folder DEL ME
        # dataset_representants_folder = Config.DataPaths.UniqueRimsCollageDatasetClipped
        # safe_mkdir(dataset_representants_folder)

        # list all dirs
        types_folders = []
        for folder in os.listdir(self.dataset_folder):
            types_folders.append(folder)

        # mapping
        print(f"folders found: {types_folders}")
        self.labels_mapping = types_folders[:]   
        self.types_folders = types_folders

    def load_data(self):
        print("Loading data")
        # prepare data holders
        self.train_ds = []
        self.train_labels = []
        self.val_ds = []
        self.val_labels = []    
        self.class_weights = dict()

        # load files from all folders containing individual rim types
        for label in self.types_folders:
            print(f"Processing folder {label}")
            safe_mkdir_clean(Config.DataPaths.WheelClassificationAugmentation + label + "/" )

            # load images of one rim type and shuffle them
            files = os.listdir(self.dataset_folder + label + '/')
            random.shuffle(files)
            
            # split into val and train selection (val dataset has its size certain)
            val_selection = files[:val_ds_size]
            train_selection = files[val_ds_size:(val_ds_size+train_ds_size)]

            # case where label does not have full count of samples
            if self.only_labels_with_full_count:
                if len(train_selection) < train_ds_size:
                    print(f"Label {label} removed because it had less than {train_ds_size} samples (had {len(train_selection)})")
                    self.labels_mapping.remove(label)
                    continue

            class_weight = float(self.train_ds_size + self.train_ds_augmented) / (len(train_selection) + self.train_ds_augmented)
            self.class_weights[label] = class_weight

            print(f"Label {label} has train_selection: {len(train_selection)} + train_aug: {self.train_ds_augmented}, val_selection: {len(val_selection)}, class_weight: {class_weight}")

            # load training selection
            for f in train_selection:
                img = cv2.imread(self.dataset_folder + label + '/' + f)
                self.save_prepared_dataset(img, label, 'train')
                # img = img/255.0
                
                self.train_ds.append(img)
                self.train_labels.append(label)

            # create augmented data by random choice from training selection
            for i in range(train_ds_augmented):
                random_file_from_train = random.choice(train_selection)
                random_img = cv2.imread(self.dataset_folder + label + '/' + random_file_from_train)
                transformed_image = self.augment_image(random_img)
                self.save_prepared_dataset(transformed_image, label, 'augmentation')
                # transformed_image = transformed_image/255.0

                self.train_ds.append(transformed_image)
                self.train_labels.append(label)

            # load val selection
            for f in val_selection:
                img = cv2.imread(self.dataset_folder + label + '/' + f)
                self.save_prepared_dataset(img, label, 'validation')
                # img = img/255.0

                self.val_ds.append(img)
                self.val_labels.append(label)

    def remap_labels(self):
        print(f"Remapping labels")
        print(f"{str(list(zip(range(len(self.labels_mapping)), self.labels_mapping)))}")
        print(f"class weights: {self.class_weights}")
        for i in range(len(self.train_labels)):
            self.train_labels[i] = self.map_label_to_sequence(self.train_labels[i])

        for i in range(len(self.val_labels)):
            self.val_labels[i] = self.map_label_to_sequence(self.val_labels[i])

        remapped_class_weights = dict()
        for key, value in self.class_weights.items():
            remapped_class_weights[self.map_label_to_sequence(key)] = value
        self.class_weights = remapped_class_weights
        print(f"remapped_class_weights: {self.class_weights}")

    def shuffle_train(self):
        # reshuffle dataset to mix types in the list 
        print("Shuffling dataset")
        zip_train = list(zip(self.train_ds, self.train_labels))
        random.shuffle(zip_train)
        self.train_ds, self.train_labels = zip(*zip_train)    

    def convert_data_to_np_arrays(self):
        print("Converting to numpy arrays")
        # convert to numpy
        self.train_ds = np.array(self.train_ds)
        self.train_labels = np.array(self.train_labels)

        self.val_ds = np.array(self.val_ds)
        self.val_labels = np.array(self.val_labels)

        np.reshape(self.train_ds, (-1,256,256,3))
        np.reshape(self.val_ds, (-1,256,256,3))

    def compute_number_of_labels(self):
        self.labels_count = len(self.labels_mapping)
        print(f"num_classes: {self.labels_count}, and labels_mapping is {self.labels_mapping}")

    def save_prepared_dataset(self, image, label, dataset_type):
        if not hasattr(self, 'saving_counter'):
            self.saving_counter = 0
        self.saving_counter = self.saving_counter + 1

        saved_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        saved_image = Image.fromarray(saved_image)
        saved_image.save(Config.DataPaths.WheelClassificationAugmentation + label + "/" + f"{dataset_type}_{label}_{self.saving_counter}.png")

    def augment_image(self, image):
        transformed_obj = self.transform(image=image)
        transformed_image = transformed_obj["image"]
        return transformed_image

    def map_label_to_sequence(self, label):
        return self.labels_mapping.index(label)

    def map_sequence_to_label(self, sequence_number):
        return self.labels_mapping[sequence_number]

class ClassificationModel:
    # model.load_weights("./model/rims_classification_checkpoints/checkpoint")
    # model.save("./model/rims_classification_checkpoints2/checkpoint2")

    def init(self, model_name, weights, layers_to_unfreeze, dataset, epochs, batch_size, saving_callback):
        print("Model inicialization")
        print(f"parameters - model_name: {model_name}, weights: {weights}, layers_to_unfreeze: {layers_to_unfreeze}")
        print(f"epochs: {epochs}, batch_size: {batch_size}, saving_callback: {saving_callback}")
        self.model_name = model_name
        self.weights = weights
        self.layers_to_unfreeze = layers_to_unfreeze
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.saving_callback = saving_callback

        self.init_model()

    def init_model(self):
        self.model = tf.keras.Sequential()

        # init base model
        base_model = self.select_model(self.model_name)


        # unfreeze layers
        print(f"Freezing layers, total_layers: {len(base_model.layers)}, freezed: {len(base_model.layers[:-self.layers_to_unfreeze])}, unfreezed: {len(base_model.layers[-self.layers_to_unfreeze:])}")
        if self.layers_to_unfreeze > 0:
            for layer in base_model.layers[:-self.layers_to_unfreeze]:
                layer.trainable = False

            for layer in base_model.layers[-self.layers_to_unfreeze:]:
                layer.trainable = True
        elif self.layers_to_unfreeze == 0:
            for layer in base_model.layers:
                layer.trainable = False
        
        # print base model summary
        base_model.summary()

        # create the rest of the model
        self.model.add(base_model)
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.Dense(classification_dataset.labels_count, activation='softmax'))

        self.model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                        optimizer='adam', 
                        metrics=['accuracy'])

        # Create a callback that saves the model's weights
        self.model_save_name = f"{self.model_name}_weights-{self.weights}_unfreezed-{self.layers_to_unfreeze}-layers"
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=f"./model/rims_classification_checkpoints/{self.model_save_name}",
                                                        monitor='val_accuracy',
                                                        mode='max',
                                                        save_best_only=True,
                                                        verbose=1)

    def select_model(self, model_name):
        img_shape = (256, 256, 3)
        # ResNet50
        # EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
        # EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, EfficientNetV2B3, EfficientNetV2B4, EfficientNetV2B5, EfficientNetV2B6, EfficientNetV2B7
        if model_name == 'ResNet50':
            return ResNet50(include_top=False, input_shape=img_shape, weights = self.weights)
        elif model_name == 'EfficientNetB0':
            return tf.keras.applications.efficientnet.EfficientNetB0(include_top=False, input_shape=img_shape, weights = self.weights)
        elif model_name == 'EfficientNetB1':
            return tf.keras.applications.efficientnet.EfficientNetB1(include_top=False, input_shape=img_shape, weights = self.weights)
        elif model_name == 'EfficientNetB2':
            return tf.keras.applications.efficientnet.EfficientNetB2(include_top=False, input_shape=img_shape, weights = self.weights)
        elif model_name == 'EfficientNetB3':
            return tf.keras.applications.efficientnet.EfficientNetB3(include_top=False, input_shape=img_shape, weights = self.weights)
        elif model_name == 'EfficientNetB4':
            return tf.keras.applications.efficientnet.EfficientNetB4(include_top=False, input_shape=img_shape, weights = self.weights)
        elif model_name == 'EfficientNetB5':
            return tf.keras.applications.efficientnet.EfficientNetB5(include_top=False, input_shape=img_shape, weights = self.weights)
        elif model_name == 'EfficientNetB6':
            return tf.keras.applications.efficientnet.EfficientNetB6(include_top=False, input_shape=img_shape, weights = self.weights)
        elif model_name == 'EfficientNetB7':
            return tf.keras.applications.efficientnet.EfficientNetB7(include_top=False, input_shape=img_shape, weights = self.weights)
        # elif model_name == 'EfficientNetV2B0':
        #     return tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False, input_shape=img_shape, weights = self.weights)
        # elif model_name == 'EfficientNetV2B1':
        #     return tf.keras.applications.efficientnet_v2.EfficientNetV2B1(include_top=False, input_shape=img_shape, weights = self.weights)
        # elif model_name == 'EfficientNetV2B2':
        #     return tf.keras.applications.efficientnet_v2.EfficientNetV2B2(include_top=False, input_shape=img_shape, weights = self.weights)
        # elif model_name == 'EfficientNetV2B3':
        #     return tf.keras.applications.efficientnet_v2.EfficientNetV2B3(include_top=False, input_shape=img_shape, weights = self.weights)
        # elif model_name == 'EfficientNetV2B4':
        #     return tf.keras.applications.efficientnet_v2.EfficientNetV2B4(include_top=False, input_shape=img_shape, weights = self.weights)
        # elif model_name == 'EfficientNetV2B5':
        #     return tf.keras.applications.efficientnet_v2.EfficientNetV2B5(include_top=False, input_shape=img_shape, weights = self.weights)
        # elif model_name == 'EfficientNetV2B6':
        #     return tf.keras.applications.efficientnet_v2.EfficientNetV2B6(include_top=False, input_shape=img_shape, weights = self.weights)
        # elif model_name == 'EfficientNetV2B7':
        #     return tf.keras.applications.efficientnet_v2.EfficientNetV2B7(include_top=False, input_shape=img_shape, weights = self.weights)
        

    def fit(self):
        callbacks = []
        if self.saving_callback:
            callbacks.append(self.cp_callback)

        # self.dataset.train_ds = self.dataset.train_ds * 255.0
        # self.dataset.val_ds = self.dataset.val_ds * 255.0

        print("Training started")
        start = time.time()
        history = self.model.fit(
        self.dataset.train_ds,
        self.dataset.train_labels,
        validation_data = (self.dataset.val_ds, self.dataset.val_labels),
        epochs=epochs,
        batch_size= batch_size,
        callbacks=callbacks,
        class_weight=self.dataset.class_weights
        )
        end = time.time()
        print("Training ended")
        print(f"Training time was: {end - start} seconds")
        print("Epochs")
        print(f"{history.epoch}")
        print("Accuracy (from history)")
        print(f"{history.history['accuracy']}")
        print(f"Maximum acc was {max(history.history['accuracy'])}")
        print("Validation Accuracy (from history)")
        print(f"{history.history['val_accuracy']}")
        print(f"Maximum val acc was {max(history.history['val_accuracy'])}")

        mpl.rcParams.update(mpl.rcParamsDefault)
        plt.cla() 
        plt.clf()

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title(self.model_save_name)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig(f"./model/rims_classification_checkpoints/plot_train_val_{self.model_save_name}.png")
        
        time.sleep(1)
        
        plt.cla() 
        plt.clf()
        pass

    def generate_confusion_matrix(self):
        val_pred = self.model.predict(self.dataset.val_ds)
        np.set_printoptions(precision=2)
        confusion_matrix = sklearn.metrics.confusion_matrix(self.dataset.val_labels, val_pred.argmax(axis=1), normalize='true')
        print(f"val_labels: {self.dataset.val_labels}")
        print(f"val_pred:   {val_pred}")
        print(f"printed as zip (gold, pred)")
        for gold, pred in list(zip(self.dataset.val_labels, val_pred.argmax(axis=1))):
            print(f"({gold}, {pred})")
        print(f"confusion_matrix:")
        print(f"{confusion_matrix}")
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=  list(map(lambda x: x if x!='unrecognized' else 'oth', self.dataset.labels_mapping)))
        fig, ax = plt.subplots(figsize=(15,15))
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                ax.text(j, i, format(confusion_matrix[i, j], '.2g'),
                        ha="center", va="center",
                        color="white" if confusion_matrix[i, j] > 0.5 else "black")
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax)
        disp.ax_.set_title(f"{self.model_save_name} confusion matrix")
        disp.figure_.savefig(f"./model/rims_classification_checkpoints/cm_{self.model_save_name}.png",dpi=300)

        time.sleep(1)
        disp.figure_.clf()

        pass

    def predict(self, image):
        prediction_probs = self.model.predict(image)
        return prediction_probs

class HogModel():
    
    def train(self, x_train, y_train, class_weight):
        self.model = RandomForestClassifier(verbose=1, class_weight=class_weight, n_estimators=300) # class_weight=class_weight,
        self.model.fit(x_train, y_train)
        pass 

    def val(self, x_test, y_test, labels_mapping):
        accuracy = self.model.score(x_test, y_test)
        predictions = self.model.predict(x_test)

        confusion_matrix = sklearn.metrics.confusion_matrix(y_test, predictions, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=  list(map(lambda x: x if x!='unrecognized' else 'oth', labels_mapping)))
        fig, ax = plt.subplots(figsize=(15,15))
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                ax.text(j, i, format(confusion_matrix[i, j], '.2g'),
                        ha="center", va="center",
                        color="white" if confusion_matrix[i, j] > 0.5 else "black")
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax)
        disp.ax_.set_title(f"Confusion matrix")
        disp.figure_.savefig(f"./model/rims_classification_checkpoints/cm_svm.png",dpi=300)
        pass


if __name__ == "__main__":
    
    #HOG
    # classification_dataset = ClassificationDataset()
    # classification_dataset.init(dataset_choice, only_labels_with_full_count, train_ds_size, train_ds_augmented, val_ds_size)
    # classification_dataset.load_data()

    # tr_ds = list(map(lambda x: hog(x, multichannel=True) ,classification_dataset.train_ds))
    # val_ds = list(map(lambda x: hog(x, multichannel=True) ,classification_dataset.val_ds))

    # hog_model = HogModel()
    # hog_model.train(tr_ds, classification_dataset.train_labels, classification_dataset.class_weights)
    # hog_model.val(val_ds, classification_dataset.val_labels, classification_dataset.labels_mapping)

    # EFFICIENT NET
    dataset_output_filename = f"dataset_creation_output_ds-{dataset_choice}_easy-{str(only_labels_with_full_count)}_tr-{str(train_ds_size)}_tra-{str(train_ds_augmented)}_val-{str(val_ds_size)}.txt"
    sys.stdout = open(f"./model/rims_classification_checkpoints/{dataset_output_filename}", "w")

    classification_dataset = ClassificationDataset()
    classification_dataset.init(dataset_choice, only_labels_with_full_count, train_ds_size, train_ds_augmented, val_ds_size)
    classification_dataset.load_data()
    classification_dataset.remap_labels()
    classification_dataset.shuffle_train()
    classification_dataset.convert_data_to_np_arrays()
    classification_dataset.compute_number_of_labels()


    model_name = 'EfficientNetB0' # 
    weights = 'imagenet' # None, 'imagenet'
    layers_to_unfreeze = 25
    epochs=30
    batch_size = 8
    saving_callback = True

    model_output_filename = f"model_train_output_{model_name}_w-{str(weights)}_unfr-{str(layers_to_unfreeze)}_ep-{str(epochs)}_cb-{str(saving_callback)}.txt"
    sys.stdout = open(f"./model/rims_classification_checkpoints/{model_output_filename}", "w")

    classification_model = ClassificationModel()
    classification_model.init(model_name, weights, layers_to_unfreeze, classification_dataset, epochs, batch_size, saving_callback)
    classification_model.fit()
    classification_model.generate_confusion_matrix()

    # model_name = 'EfficientNetB0' # 
    # weights = 'imagenet' # None, 'imagenet'
    # layers_to_unfreeze = 1
    # epochs=20
    # batch_size = 8
    # saving_callback = False

    # model_output_filename = f"model_train_output_{model_name}_w-{str(weights)}_unfr-{str(layers_to_unfreeze)}_ep-{str(epochs)}_cb-{str(saving_callback)}.txt"
    # sys.stdout = open(f"./model/rims_classification_checkpoints/{model_output_filename}", "w")

    # classification_model = ClassificationModel()
    # classification_model.init(model_name, weights, layers_to_unfreeze, classification_dataset, epochs, batch_size, saving_callback)
    # classification_model.fit()
    # classification_model.generate_confusion_matrix()

    # model_name = 'EfficientNetB0' # 
    # weights = 'imagenet' # None, 'imagenet'
    # layers_to_unfreeze = 2
    # epochs=20
    # batch_size = 8
    # saving_callback = False

    # model_output_filename = f"model_train_output_{model_name}_w-{str(weights)}_unfr-{str(layers_to_unfreeze)}_ep-{str(epochs)}_cb-{str(saving_callback)}.txt"
    # sys.stdout = open(f"./model/rims_classification_checkpoints/{model_output_filename}", "w")

    # classification_model = ClassificationModel()
    # classification_model.init(model_name, weights, layers_to_unfreeze, classification_dataset, epochs, batch_size, saving_callback)
    # classification_model.fit()
    # classification_model.generate_confusion_matrix()

    # model_name = 'EfficientNetB0' # 
    # weights = 'imagenet' # None, 'imagenet'
    # layers_to_unfreeze = 3
    # epochs=20
    # batch_size = 8
    # saving_callback = False

    # model_output_filename = f"model_train_output_{model_name}_w-{str(weights)}_unfr-{str(layers_to_unfreeze)}_ep-{str(epochs)}_cb-{str(saving_callback)}.txt"
    # sys.stdout = open(f"./model/rims_classification_checkpoints/{model_output_filename}", "w")

    # classification_model = ClassificationModel()
    # classification_model.init(model_name, weights, layers_to_unfreeze, classification_dataset, epochs, batch_size, saving_callback)
    # classification_model.fit()
    # classification_model.generate_confusion_matrix()

    # model_name = 'EfficientNetB0' # 
    # weights = 'imagenet' # None, 'imagenet'
    # layers_to_unfreeze = 5
    # epochs=20
    # batch_size = 8
    # saving_callback = False

    # model_output_filename = f"model_train_output_{model_name}_w-{str(weights)}_unfr-{str(layers_to_unfreeze)}_ep-{str(epochs)}_cb-{str(saving_callback)}.txt"
    # sys.stdout = open(f"./model/rims_classification_checkpoints/{model_output_filename}", "w")

    # classification_model = ClassificationModel()
    # classification_model.init(model_name, weights, layers_to_unfreeze, classification_dataset, epochs, batch_size, saving_callback)
    # classification_model.fit()
    # classification_model.generate_confusion_matrix()

    # model_name = 'EfficientNetB0' # 
    # weights = 'imagenet' # None, 'imagenet'
    # layers_to_unfreeze = 10
    # epochs=20
    # batch_size = 8
    # saving_callback = False

    # model_output_filename = f"model_train_output_{model_name}_w-{str(weights)}_unfr-{str(layers_to_unfreeze)}_ep-{str(epochs)}_cb-{str(saving_callback)}.txt"
    # sys.stdout = open(f"./model/rims_classification_checkpoints/{model_output_filename}", "w")

    # classification_model = ClassificationModel()
    # classification_model.init(model_name, weights, layers_to_unfreeze, classification_dataset, epochs, batch_size, saving_callback)
    # classification_model.fit()
    # classification_model.generate_confusion_matrix()

    # model_name = 'EfficientNetB0' # 
    # weights = 'imagenet' # None, 'imagenet'
    # layers_to_unfreeze = 25
    # epochs=20
    # batch_size = 8
    # saving_callback = False

    # model_output_filename = f"model_train_output_{model_name}_w-{str(weights)}_unfr-{str(layers_to_unfreeze)}_ep-{str(epochs)}_cb-{str(saving_callback)}.txt"
    # sys.stdout = open(f"./model/rims_classification_checkpoints/{model_output_filename}", "w")

    # classification_model = ClassificationModel()
    # classification_model.init(model_name, weights, layers_to_unfreeze, classification_dataset, epochs, batch_size, saving_callback)
    # classification_model.fit()
    # classification_model.generate_confusion_matrix()

    # model_name = 'EfficientNetB0' # 
    # weights = 'imagenet' # None, 'imagenet'
    # layers_to_unfreeze = 50
    # epochs=20
    # batch_size = 8
    # saving_callback = False

    # model_output_filename = f"model_train_output_{model_name}_w-{str(weights)}_unfr-{str(layers_to_unfreeze)}_ep-{str(epochs)}_cb-{str(saving_callback)}.txt"
    # sys.stdout = open(f"./model/rims_classification_checkpoints/{model_output_filename}", "w")

    # classification_model = ClassificationModel()
    # classification_model.init(model_name, weights, layers_to_unfreeze, classification_dataset, epochs, batch_size, saving_callback)
    # classification_model.fit()
    # classification_model.generate_confusion_matrix()

    # model_name = 'EfficientNetB0' # 
    # weights = 'imagenet' # None, 'imagenet'
    # layers_to_unfreeze = 100
    # epochs=20
    # batch_size = 8
    # saving_callback = False

    # model_output_filename = f"model_train_output_{model_name}_w-{str(weights)}_unfr-{str(layers_to_unfreeze)}_ep-{str(epochs)}_cb-{str(saving_callback)}.txt"
    # sys.stdout = open(f"./model/rims_classification_checkpoints/{model_output_filename}", "w")

    # classification_model = ClassificationModel()
    # classification_model.init(model_name, weights, layers_to_unfreeze, classification_dataset, epochs, batch_size, saving_callback)
    # classification_model.fit()
    # classification_model.generate_confusion_matrix()

    # model_name = 'EfficientNetB0' # 
    # weights = 'imagenet' # None, 'imagenet'
    # layers_to_unfreeze = 237
    # epochs=20
    # batch_size = 8
    # saving_callback = False

    # model_output_filename = f"model_train_output_{model_name}_w-{str(weights)}_unfr-{str(layers_to_unfreeze)}_ep-{str(epochs)}_cb-{str(saving_callback)}.txt"
    # sys.stdout = open(f"./model/rims_classification_checkpoints/{model_output_filename}", "w")

    # classification_model = ClassificationModel()
    # classification_model.init(model_name, weights, layers_to_unfreeze, classification_dataset, epochs, batch_size, saving_callback)
    # classification_model.fit()
    # classification_model.generate_confusion_matrix()



