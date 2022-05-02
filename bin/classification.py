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
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

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
# train_ds_augmented = 0
# dataset_choice = 'dataset21'
# dataset_choice = 'dataset21'  'dataset31'

# model configuration
# ResNet50
# EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
# EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2, EfficientNetV2B3, EfficientNetV2B4, EfficientNetV2B5, EfficientNetV2B6, EfficientNetV2B7
# model_name = 'EfficientNetB0' # 
# weights = 'imagenet' # None, 'imagenet'
# layers_to_unfreeze = 0
# epochs=2
# batch_size = 8
# saving_callback = False

class ClassificationDataset:

    def init(self, dataset_choice, train_ds_augmented):
        print("Dataset initialization")
        print(f"parameters - dataset_choice: {dataset_choice}")
        print(f"train_ds_augmented: {train_ds_augmented}")

        self.dataset_choice = dataset_choice
        self.train_ds_augmented = train_ds_augmented
        self.dataset_folder = Config.DataPaths.Dataset31

        self.init_augmentation()
        self.init_labels_mapping()
        safe_mkdir_clean(Config.DataPaths.WheelClassificationAugmentation)

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
        types_folders = []
        if self.dataset_choice == 'dataset21':
            types_folders = ['10', '30', '40', '50', '60', '70', '71', '80', '90', '91', '100', '101', '110', '120', '130', '141', '151', '160', '181', '190', '200']
        elif self.dataset_choice == 'dataset31':
            # list all dirs
            types_folders = []
            for folder in os.listdir(self.dataset_folder):
                types_folders.append(folder)

        # mapping
        print(f"folders used: {types_folders}")
        self.labels_mapping = types_folders[:]   
        self.types_folders = types_folders

    def load_data(self):
        print("Loading data", flush=True)
        # prepare data holders
        self.train_ds = []
        self.train_labels = []
        self.val_ds = []
        self.val_labels = []    
        self.test_ds = []
        self.test_labels = []    
        self.class_weights = dict()

        # load files from all folders containing individual rim types
        for label in self.types_folders:
            print(f"Processing folder {label}", flush=True)
            safe_mkdir_clean(Config.DataPaths.WheelClassificationAugmentation + label + "/" )

            train_files_folder = self.dataset_folder + f"{label}/train/" 
            val_files_folder = self.dataset_folder + f"{label}/val/"
            test_files_folder = self.dataset_folder + f"{label}/test/"

            train_files = os.listdir(train_files_folder)
            val_files = os.listdir(val_files_folder)
            test_files = os.listdir(test_files_folder)

            random.shuffle(train_files)
            random.shuffle(val_files)
            random.shuffle(test_files)

            class_weight = float(100 + self.train_ds_augmented) / (len(train_files) + self.train_ds_augmented)
            self.class_weights[label] = class_weight

            print(f"Label {label} has train: {len(train_files)} + train_aug: {self.train_ds_augmented}, val: {len(val_files)}, test: {len(test_files)} class_weight: {class_weight}")

            # load training selection
            for f in train_files:
                img = cv2.imread(train_files_folder + f)
                self.save_prepared_dataset(img, label, 'train')
                
                self.train_ds.append(img)
                self.train_labels.append(label)

            # create augmented data by random choice from training selection
            for i in range(train_ds_augmented):
                random_file_from_train = random.choice(train_files)
                random_img = cv2.imread(train_files_folder + random_file_from_train)
                transformed_image = self.augment_image(random_img)
                self.save_prepared_dataset(transformed_image, label, 'augmentation')

                self.train_ds.append(transformed_image)
                self.train_labels.append(label)

            # load val selection
            for f in val_files:
                img = cv2.imread(val_files_folder + f)
                self.save_prepared_dataset(img, label, 'validation')

                self.val_ds.append(img)
                self.val_labels.append(label)

            # load test selection
            for f in test_files:
                img = cv2.imread(test_files_folder + f)
                self.save_prepared_dataset(img, label, 'test')
                self.test_ds.append(img)
                self.test_labels.append(label)

        print(f"Length of the data is: train: {str(len(self.train_ds))}, val: {str(len(self.val_ds))}, test: {str(len(self.test_ds))}")
        print("Loading of the data complete", flush=True)

    def remap_labels(self):
        print(f"Remapping labels")
        print(f"{str(list(zip(range(len(self.labels_mapping)), self.labels_mapping)))}")
        print(f"class weights: {self.class_weights}")
        for i in range(len(self.train_labels)):
            self.train_labels[i] = self.map_label_to_sequence(self.train_labels[i])

        for i in range(len(self.val_labels)):
            self.val_labels[i] = self.map_label_to_sequence(self.val_labels[i])

        for i in range(len(self.test_labels)):
            self.test_labels[i] = self.map_label_to_sequence(self.test_labels[i])

        remapped_class_weights = dict()
        for key, value in self.class_weights.items():
            remapped_class_weights[self.map_label_to_sequence(key)] = value
        self.class_weights = remapped_class_weights
        print(f"remapped_class_weights: {self.class_weights}")

    def shuffle_train(self):
        # reshuffle dataset to mix types in the list 
        print("Shuffling dataset", flush=True)
        zip_train = list(zip(self.train_ds, self.train_labels))
        random.shuffle(zip_train)
        self.train_ds, self.train_labels = zip(*zip_train)    

    def convert_data_to_np_arrays(self):
        print("Converting to numpy arrays", flush=True)
        # convert to numpy
        self.train_ds = np.array(self.train_ds)
        self.train_labels = np.array(self.train_labels)

        self.val_ds = np.array(self.val_ds)
        self.val_labels = np.array(self.val_labels)

        self.test_ds = np.array(self.test_ds)
        self.test_labels = np.array(self.test_labels)

        np.reshape(self.train_ds, (-1,256,256,3))
        np.reshape(self.val_ds, (-1,256,256,3))
        np.reshape(self.test_ds, (-1,256,256,3))

    def compute_number_of_labels(self):
        self.labels_count = len(self.labels_mapping)
        print(f"num_classes: {self.labels_count}, and labels_mapping is {self.labels_mapping}", flush=True)

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

        print("Training started", flush=True)
        start = time.time()
        history = self.model.fit(
        self.dataset.train_ds,
        self.dataset.train_labels,
        validation_data = (self.dataset.val_ds, self.dataset.val_labels),
        epochs=self.epochs,
        batch_size= self.batch_size,
        callbacks=callbacks,
        class_weight=self.dataset.class_weights
        )
        end = time.time()
        print("Training ended", flush=True)
        print(f"Training time was: {end - start} seconds")
        print("Epochs")
        print(f"{history.epoch}")
        print("Accuracy (from history)")
        print(f"{history.history['accuracy']}")
        print(f"Maximum acc was {max(history.history['accuracy'])}")
        print("Validation Accuracy (from history)")
        print(f"{history.history['val_accuracy']}")
        print(f"Maximum val acc was {max(history.history['val_accuracy'])}", flush=True)

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
        val_pred_argmax = val_pred.argmax(axis=1)
        test_pred = self.model.predict(self.dataset.test_ds)
        test_pred_argmax = test_pred.argmax(axis=1)

        val_accuracy = accuracy_score(self.dataset.val_labels, val_pred_argmax)
        val_balanced_accuracy = balanced_accuracy_score(self.dataset.val_labels, val_pred_argmax)
        test_accuracy = accuracy_score(self.dataset.test_labels, test_pred_argmax)
        test_balanced_accuracy = balanced_accuracy_score(self.dataset.test_labels, test_pred_argmax)
        print(f"val_acc = {val_accuracy}, val_balanced_acc = {val_balanced_accuracy}")
        print(f"test_acc = {test_accuracy}, test_balanced_acc = {test_balanced_accuracy}", flush=True)

        np.set_printoptions(precision=2)
        confusion_matrix = sklearn.metrics.confusion_matrix(self.dataset.val_labels, val_pred_argmax, normalize='true')
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

    def pred_speed(self):
        files_path = "./data/dataset31/10/val/"
        files = os.listdir(files_path)
        print("Measuring pred speed of EffNet:")
        for file in files:
            img = cv2.imread(files_path + file)
            img = np.array([img])
            np.reshape(img, (-1,256,256,3))

            start = time.time()
            self.model.predict(img)
            end = time.time()
            print(f"inferece_time: {end - start}")

    def test(self, classification_dataset, model_checkpoint_path):
        model = keras.models.load_model(model_checkpoint_path)

        prediction_probs = model.predict(classification_dataset.test_ds)
        predictions = prediction_probs.argmax(axis=-1)

        test_accuracy = accuracy_score(classification_dataset.test_labels, predictions)
        test_balanced_accuracy = balanced_accuracy_score(classification_dataset.test_labels, predictions)
        print(f"test_acc = {test_accuracy}, test_balanced_acc = {test_balanced_accuracy}", flush=True)



class HogModel():
    
    def init_parameters(self, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), multichannel=None, ml_alg='logreg'):
        # skimage - hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys', visualize=False, transform_sqrt=False, feature_vector=True, multichannel=None, *, channel_axis=None)
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.multichannel = multichannel
        self.ml_alg = ml_alg

        print(f"HOG init params: orientations: {orientations}, pixels_per_cell: {pixels_per_cell}, cells_per_block: {cells_per_block}, ml_model: {ml_alg}", flush=True)

    def init_data(self, train_data, val_data, test_data):
        start = time.time()
        self.train_data = list(map(lambda x: self.custom_params_hog(x) ,train_data))
        self.val_data = list(map(lambda x: self.custom_params_hog(x) ,val_data))
        self.test_data = list(map(lambda x: self.custom_params_hog(x) ,test_data))
        end = time.time()
        print(f"hog making features from the data time: {end - start}")

    def custom_params_hog(self, image):
        return hog(image, multichannel=True, orientations=self.orientations, pixels_per_cell=self.pixels_per_cell, cells_per_block=self.cells_per_block) 

    def train(self, y_train, class_weight):
        print("HOG train")
        if self.ml_alg == 'logreg':
            self.model = LogisticRegression(class_weight=class_weight)
        elif self.ml_alg == 'linsvc':
            self.model = LinearSVC(class_weight=class_weight)
        start = time.time()
        self.model.fit(self.train_data, y_train)
        end = time.time()
        print(f"train_time: {end - start}")
        pass 

    def val(self, y_val, labels_mapping, y_test):
        print("HOG computing scores")
        predictions_val = self.model.predict(self.val_data)
        predictions_test = self.model.predict(self.test_data)
        accuracy = self.model.score(self.val_data, y_val)

        val_accuracy = accuracy_score(y_val, predictions_val)
        val_balanced_accuracy = balanced_accuracy_score(y_val, predictions_val)
        test_accuracy = accuracy_score(y_test, predictions_test)
        test_balanced_accuracy = balanced_accuracy_score(y_test, predictions_test)
        print(f"accuracy acording to the model: {accuracy}")
        print(f"val_acc = {val_accuracy}, val_balanced_acc = {val_balanced_accuracy}")
        print(f"test_acc = {test_accuracy}, test_balanced_acc = {test_balanced_accuracy}", flush=True)

        description_string = f"or-{str(self.orientations)}_ppc-{str(self.pixels_per_cell)}_cpb-{str(self.cells_per_block)}_mlalg-{str(self.ml_alg)}"

        confusion_matrix = sklearn.metrics.confusion_matrix(y_val, predictions_val, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=  list(map(lambda x: x if x!='unrecognized' else 'oth', labels_mapping)))
        fig, ax = plt.subplots(figsize=(15,15))
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                ax.text(j, i, format(confusion_matrix[i, j], '.2g'),
                        ha="center", va="center",
                        color="white" if confusion_matrix[i, j] > 0.5 else "black")
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax)
        disp.ax_.set_title(f"Confusion matrix")
        disp.figure_.savefig(f"./model/rims_classification_checkpoints/cm_hogsvm_val_{description_string}.png",dpi=300)
        
        confusion_matrix = sklearn.metrics.confusion_matrix(y_test, predictions_test, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=  list(map(lambda x: x if x!='unrecognized' else 'oth', labels_mapping)))
        fig, ax = plt.subplots(figsize=(15,15))
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                ax.text(j, i, format(confusion_matrix[i, j], '.2g'),
                        ha="center", va="center",
                        color="white" if confusion_matrix[i, j] > 0.5 else "black")
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax)
        disp.ax_.set_title(f"Confusion matrix")
        disp.figure_.savefig(f"./model/rims_classification_checkpoints/cm_hogsvm_test_{description_string}.png",dpi=300)
        pass

    def pred_speed(self):
        files_path = "./data/dataset31/10/val/"
        files = os.listdir(files_path)
        print(f"Measuring pred speed of HOG+{self.ml_alg}:")
        for file in files:
            img = cv2.imread(files_path + file)

            print(f"size of image is[bytes]: {sys.getsizeof(img)}: ")
            start = time.time()
            img = list(map(lambda x: self.custom_params_hog(x), [img]))
            end = time.time()
            print(f"conversion to hog features: {end - start}")
            print(f"size of hog features is[bytes]: {sys.getsizeof(img[0])}, and its lengths is: {str(len(img[0]))}: ")

            start = time.time()
            self.model.predict(img)
            end = time.time()
            print(f"inferece_time: {end - start}")


if __name__ == "__main__":
    
    ###########################################################################################################################

    # #HOG
    # sys.stdout = open(f"./model/rims_classification_checkpoints/hog_svm_output.txt", "w")

    # train_ds_augmented = 0
    # dataset_choice = 'dataset21'

    # classification_dataset = ClassificationDataset()
    # classification_dataset.init(dataset_choice, train_ds_augmented)
    # classification_dataset.load_data()
    # classification_dataset.shuffle_train()

    # ###

    # multichannel = True
    # ml_alg = 'linsvc'

    # orientations=9
    # pixels_per_cell=(4, 4) 
    # cells_per_block=(3, 3)

    # hog_model = HogModel()
    # hog_model.init_parameters(orientations, pixels_per_cell, cells_per_block, multichannel, ml_alg)
    # hog_model.init_data(classification_dataset.train_ds, classification_dataset.val_ds, classification_dataset.test_ds)
    # hog_model.train(classification_dataset.train_labels, classification_dataset.class_weights)
    # hog_model.val(classification_dataset.val_labels, classification_dataset.labels_mapping, classification_dataset.test_labels)
    # hog_model.pred_speed()

    # ###

    # orientations=9
    # pixels_per_cell=(8, 8) 
    # cells_per_block=(3, 3)

    # hog_model = HogModel()
    # hog_model.init_parameters(orientations, pixels_per_cell, cells_per_block, multichannel, ml_alg)
    # hog_model.init_data(classification_dataset.train_ds, classification_dataset.val_ds, classification_dataset.test_ds)
    # hog_model.train(classification_dataset.train_labels, classification_dataset.class_weights)
    # hog_model.val(classification_dataset.val_labels, classification_dataset.labels_mapping, classification_dataset.test_labels)
    # hog_model.pred_speed()

    # ###

    # orientations=9
    # pixels_per_cell=(16, 16) 
    # cells_per_block=(3, 3)

    # hog_model = HogModel()
    # hog_model.init_parameters(orientations, pixels_per_cell, cells_per_block, multichannel, ml_alg)
    # hog_model.init_data(classification_dataset.train_ds, classification_dataset.val_ds, classification_dataset.test_ds)
    # hog_model.train(classification_dataset.train_labels, classification_dataset.class_weights)
    # hog_model.val(classification_dataset.val_labels, classification_dataset.labels_mapping, classification_dataset.test_labels)
    # hog_model.pred_speed()

    # ###

    # orientations=13
    # pixels_per_cell=(4, 4) 
    # cells_per_block=(3, 3)

    # hog_model = HogModel()
    # hog_model.init_parameters(orientations, pixels_per_cell, cells_per_block, multichannel, ml_alg)
    # hog_model.init_data(classification_dataset.train_ds, classification_dataset.val_ds, classification_dataset.test_ds)
    # hog_model.train(classification_dataset.train_labels, classification_dataset.class_weights)
    # hog_model.val(classification_dataset.val_labels, classification_dataset.labels_mapping, classification_dataset.test_labels)
    # hog_model.pred_speed()

    # ###

    # orientations=13
    # pixels_per_cell=(8, 8) 
    # cells_per_block=(3, 3)

    # hog_model = HogModel()
    # hog_model.init_parameters(orientations, pixels_per_cell, cells_per_block, multichannel, ml_alg)
    # hog_model.init_data(classification_dataset.train_ds, classification_dataset.val_ds, classification_dataset.test_ds)
    # hog_model.train(classification_dataset.train_labels, classification_dataset.class_weights)
    # hog_model.val(classification_dataset.val_labels, classification_dataset.labels_mapping, classification_dataset.test_labels)
    # hog_model.pred_speed()

    # ###

    # orientations=13
    # pixels_per_cell=(16, 16) 
    # cells_per_block=(3, 3)

    # hog_model = HogModel()
    # hog_model.init_parameters(orientations, pixels_per_cell, cells_per_block, multichannel, ml_alg)
    # hog_model.init_data(classification_dataset.train_ds, classification_dataset.val_ds, classification_dataset.test_ds)
    # hog_model.train(classification_dataset.train_labels, classification_dataset.class_weights)
    # hog_model.val(classification_dataset.val_labels, classification_dataset.labels_mapping, classification_dataset.test_labels)
    # hog_model.pred_speed()

    # ###

    # orientations=16
    # pixels_per_cell=(4, 4) 
    # cells_per_block=(3, 3)

    # hog_model = HogModel()
    # hog_model.init_parameters(orientations, pixels_per_cell, cells_per_block, multichannel, ml_alg)
    # hog_model.init_data(classification_dataset.train_ds, classification_dataset.val_ds, classification_dataset.test_ds)
    # hog_model.train(classification_dataset.train_labels, classification_dataset.class_weights)
    # hog_model.val(classification_dataset.val_labels, classification_dataset.labels_mapping, classification_dataset.test_labels)
    # hog_model.pred_speed()

    # ###

    # orientations=16
    # pixels_per_cell=(8, 8) 
    # cells_per_block=(3, 3)

    # hog_model = HogModel()
    # hog_model.init_parameters(orientations, pixels_per_cell, cells_per_block, multichannel, ml_alg)
    # hog_model.init_data(classification_dataset.train_ds, classification_dataset.val_ds, classification_dataset.test_ds)
    # hog_model.train(classification_dataset.train_labels, classification_dataset.class_weights)
    # hog_model.val(classification_dataset.val_labels, classification_dataset.labels_mapping, classification_dataset.test_labels)
    # hog_model.pred_speed()

    # ###

    # orientations=16
    # pixels_per_cell=(16, 16) 
    # cells_per_block=(3, 3)

    # hog_model = HogModel()
    # hog_model.init_parameters(orientations, pixels_per_cell, cells_per_block, multichannel, ml_alg)
    # hog_model.init_data(classification_dataset.train_ds, classification_dataset.val_ds, classification_dataset.test_ds)
    # hog_model.train(classification_dataset.train_labels, classification_dataset.class_weights)
    # hog_model.val(classification_dataset.val_labels, classification_dataset.labels_mapping, classification_dataset.test_labels)
    # hog_model.pred_speed()

     # ###

    # orientations= 9
    # pixels_per_cell=(24, 24) 
    # cells_per_block=(3, 3)

    # hog_model = HogModel()
    # hog_model.init_parameters(orientations, pixels_per_cell, cells_per_block, multichannel, ml_alg)
    # hog_model.init_data(classification_dataset.train_ds, classification_dataset.val_ds, classification_dataset.test_ds)
    # hog_model.train(classification_dataset.train_labels, classification_dataset.class_weights)
    # hog_model.val(classification_dataset.val_labels, classification_dataset.labels_mapping, classification_dataset.test_labels)
    # hog_model.pred_speed()

    # ###

    # orientations=13
    # pixels_per_cell=(24, 24) 
    # cells_per_block=(3, 3)

    # hog_model = HogModel()
    # hog_model.init_parameters(orientations, pixels_per_cell, cells_per_block, multichannel, ml_alg)
    # hog_model.init_data(classification_dataset.train_ds, classification_dataset.val_ds, classification_dataset.test_ds)
    # hog_model.train(classification_dataset.train_labels, classification_dataset.class_weights)
    # hog_model.val(classification_dataset.val_labels, classification_dataset.labels_mapping, classification_dataset.test_labels)
    # hog_model.pred_speed()

    # ###

    # orientations=13
    # pixels_per_cell=(32, 32) 
    # cells_per_block=(3, 3)

    # hog_model = HogModel()
    # hog_model.init_parameters(orientations, pixels_per_cell, cells_per_block, multichannel, ml_alg)
    # hog_model.init_data(classification_dataset.train_ds, classification_dataset.val_ds, classification_dataset.test_ds)
    # hog_model.train(classification_dataset.train_labels, classification_dataset.class_weights)
    # hog_model.val(classification_dataset.val_labels, classification_dataset.labels_mapping, classification_dataset.test_labels)
    # hog_model.pred_speed()

    ###########################################################################################################################
    ### EFFICIENT NET TRAINING
    # train_ds_augmented = 0
    # dataset_choice = 'dataset31'

    # dataset_output_filename = f"dataset_creation_output_ds-{dataset_choice}_tra-{str(train_ds_augmented)}.txt"
    # sys.stdout = open(f"./model/rims_classification_checkpoints/{dataset_output_filename}", "w")

    # classification_dataset = ClassificationDataset()
    # classification_dataset.init(dataset_choice, train_ds_augmented)
    # classification_dataset.load_data()
    # classification_dataset.remap_labels()
    # classification_dataset.shuffle_train()
    # classification_dataset.convert_data_to_np_arrays()
    # classification_dataset.compute_number_of_labels()


    # model_name = 'EfficientNetB0' # 
    # weights = 'imagenet' # None, 'imagenet'
    # layers_to_unfreeze = 0
    # epochs=50
    # batch_size = 8
    # saving_callback = False

    # model_output_filename = f"model_train_output_{model_name}_w-{str(weights)}_unfr-{str(layers_to_unfreeze)}_ep-{str(epochs)}_cb-{str(saving_callback)}.txt"
    # sys.stdout = open(f"./model/rims_classification_checkpoints/{model_output_filename}", "w")

    # classification_model = ClassificationModel()
    # classification_model.init(model_name, weights, layers_to_unfreeze, classification_dataset, epochs, batch_size, saving_callback)
    # classification_model.fit()
    # classification_model.generate_confusion_matrix()
    # classification_model.pred_speed()

    # model_name = 'EfficientNetB0' # 
    # weights = 'imagenet' # None, 'imagenet'
    # layers_to_unfreeze = 1
    # epochs=50
    # batch_size = 8
    # saving_callback = False

    # model_output_filename = f"model_train_output_{model_name}_w-{str(weights)}_unfr-{str(layers_to_unfreeze)}_ep-{str(epochs)}_cb-{str(saving_callback)}.txt"
    # sys.stdout = open(f"./model/rims_classification_checkpoints/{model_output_filename}", "w")

    # classification_model = ClassificationModel()
    # classification_model.init(model_name, weights, layers_to_unfreeze, classification_dataset, epochs, batch_size, saving_callback)
    # classification_model.fit()
    # classification_model.generate_confusion_matrix()
    # classification_model.pred_speed()

    # model_name = 'EfficientNetB0' # 
    # weights = 'imagenet' # None, 'imagenet'
    # layers_to_unfreeze = 2
    # epochs=50
    # batch_size = 8
    # saving_callback = False

    # model_output_filename = f"model_train_output_{model_name}_w-{str(weights)}_unfr-{str(layers_to_unfreeze)}_ep-{str(epochs)}_cb-{str(saving_callback)}.txt"
    # sys.stdout = open(f"./model/rims_classification_checkpoints/{model_output_filename}", "w")

    # classification_model = ClassificationModel()
    # classification_model.init(model_name, weights, layers_to_unfreeze, classification_dataset, epochs, batch_size, saving_callback)
    # classification_model.fit()
    # classification_model.generate_confusion_matrix()
    # classification_model.pred_speed()

    # model_name = 'EfficientNetB0' # 
    # weights = 'imagenet' # None, 'imagenet'
    # layers_to_unfreeze = 3
    # epochs=50
    # batch_size = 8
    # saving_callback = False

    # model_output_filename = f"model_train_output_{model_name}_w-{str(weights)}_unfr-{str(layers_to_unfreeze)}_ep-{str(epochs)}_cb-{str(saving_callback)}.txt"
    # sys.stdout = open(f"./model/rims_classification_checkpoints/{model_output_filename}", "w")

    # classification_model = ClassificationModel()
    # classification_model.init(model_name, weights, layers_to_unfreeze, classification_dataset, epochs, batch_size, saving_callback)
    # classification_model.fit()
    # classification_model.generate_confusion_matrix()
    # classification_model.pred_speed()

    # model_name = 'EfficientNetB0' # 
    # weights = 'imagenet' # None, 'imagenet'
    # layers_to_unfreeze = 5
    # epochs=50
    # batch_size = 8
    # saving_callback = False

    # model_output_filename = f"model_train_output_{model_name}_w-{str(weights)}_unfr-{str(layers_to_unfreeze)}_ep-{str(epochs)}_cb-{str(saving_callback)}.txt"
    # sys.stdout = open(f"./model/rims_classification_checkpoints/{model_output_filename}", "w")

    # classification_model = ClassificationModel()
    # classification_model.init(model_name, weights, layers_to_unfreeze, classification_dataset, epochs, batch_size, saving_callback)
    # classification_model.fit()
    # classification_model.generate_confusion_matrix()
    # classification_model.pred_speed()

    # model_name = 'EfficientNetB0' # 
    # weights = 'imagenet' # None, 'imagenet'
    # layers_to_unfreeze = 10
    # epochs=50
    # batch_size = 8
    # saving_callback = False

    # model_output_filename = f"model_train_output_{model_name}_w-{str(weights)}_unfr-{str(layers_to_unfreeze)}_ep-{str(epochs)}_cb-{str(saving_callback)}.txt"
    # sys.stdout = open(f"./model/rims_classification_checkpoints/{model_output_filename}", "w")

    # classification_model = ClassificationModel()
    # classification_model.init(model_name, weights, layers_to_unfreeze, classification_dataset, epochs, batch_size, saving_callback)
    # classification_model.fit()
    # classification_model.generate_confusion_matrix()
    # classification_model.pred_speed()

    # model_name = 'EfficientNetB0' # 
    # weights = 'imagenet' # None, 'imagenet'
    # layers_to_unfreeze = 25
    # epochs=50
    # batch_size = 8
    # saving_callback = True

    # model_output_filename = f"model_train_output_{model_name}_w-{str(weights)}_unfr-{str(layers_to_unfreeze)}_ep-{str(epochs)}_cb-{str(saving_callback)}.txt"
    # sys.stdout = open(f"./model/rims_classification_checkpoints/{model_output_filename}", "w")

    # classification_model = ClassificationModel()
    # classification_model.init(model_name, weights, layers_to_unfreeze, classification_dataset, epochs, batch_size, saving_callback)
    # classification_model.fit()
    # classification_model.generate_confusion_matrix()
    # classification_model.pred_speed()

    # model_name = 'EfficientNetB0' # 
    # weights = 'imagenet' # None, 'imagenet'
    # layers_to_unfreeze = 50
    # epochs=50
    # batch_size = 8
    # saving_callback = False

    # model_output_filename = f"model_train_output_{model_name}_w-{str(weights)}_unfr-{str(layers_to_unfreeze)}_ep-{str(epochs)}_cb-{str(saving_callback)}.txt"
    # sys.stdout = open(f"./model/rims_classification_checkpoints/{model_output_filename}", "w")

    # classification_model = ClassificationModel()
    # classification_model.init(model_name, weights, layers_to_unfreeze, classification_dataset, epochs, batch_size, saving_callback)
    # classification_model.fit()
    # classification_model.generate_confusion_matrix()
    # classification_model.pred_speed()

    # model_name = 'EfficientNetB0' # 
    # weights = 'imagenet' # None, 'imagenet'
    # layers_to_unfreeze = 100
    # epochs=50
    # batch_size = 8
    # saving_callback = False

    # model_output_filename = f"model_train_output_{model_name}_w-{str(weights)}_unfr-{str(layers_to_unfreeze)}_ep-{str(epochs)}_cb-{str(saving_callback)}.txt"
    # sys.stdout = open(f"./model/rims_classification_checkpoints/{model_output_filename}", "w")

    # classification_model = ClassificationModel()
    # classification_model.init(model_name, weights, layers_to_unfreeze, classification_dataset, epochs, batch_size, saving_callback)
    # classification_model.fit()
    # classification_model.generate_confusion_matrix()
    # classification_model.pred_speed()

    # model_name = 'EfficientNetB0' # 
    # weights = 'imagenet' # None, 'imagenet'
    # layers_to_unfreeze = 237
    # epochs=50
    # batch_size = 8
    # saving_callback = False

    # model_output_filename = f"model_train_output_{model_name}_w-{str(weights)}_unfr-{str(layers_to_unfreeze)}_ep-{str(epochs)}_cb-{str(saving_callback)}.txt"
    # sys.stdout = open(f"./model/rims_classification_checkpoints/{model_output_filename}", "w")

    # classification_model = ClassificationModel()
    # classification_model.init(model_name, weights, layers_to_unfreeze, classification_dataset, epochs, batch_size, saving_callback)
    # classification_model.fit()
    # classification_model.generate_confusion_matrix()
    # classification_model.pred_speed()

    ###########################################################################################################################

    ## EFFNET TESTING
    train_ds_augmented = 0
    dataset_choice = 'dataset21'

    classification_dataset = ClassificationDataset()
    classification_dataset.init(dataset_choice, train_ds_augmented)
    classification_dataset.load_data()
    classification_dataset.remap_labels()
    classification_dataset.shuffle_train()
    classification_dataset.convert_data_to_np_arrays()
    classification_dataset.compute_number_of_labels()

    classification_model = ClassificationModel()
    classification_model.test(classification_dataset, "./model/rims_classification_checkpoints\EfficientNetB0_weights-imagenet_unfreezed-25-layers_dataset21")

    train_ds_augmented = 0
    dataset_choice = 'dataset31'

    classification_dataset = ClassificationDataset()
    classification_dataset.init(dataset_choice, train_ds_augmented)
    classification_dataset.load_data()
    classification_dataset.remap_labels()
    classification_dataset.shuffle_train()
    classification_dataset.convert_data_to_np_arrays()
    classification_dataset.compute_number_of_labels()

    classification_model = ClassificationModel()
    classification_model.test(classification_dataset, "./model/rims_classification_checkpoints\EfficientNetB0_weights-imagenet_unfreezed-25-layers_dataset31")




















# HOG vizualization code 
# img = cv2.imread("./data/crops/2019_05_13_19_24_17_B_frame557_bb0_Wheel.png")

# fd, hog_image = hog(img, visualize=True, multichannel=True, orientations=8, pixels_per_cell=(16, 16),
#                 cells_per_block=(1, 1))
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 16), sharex=True, sharey=True) 

# ax1.imshow(img, cmap=plt.cm.gray) 
# ax1.set_title('Input image') 

# # Rescale histogram for better display 
# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10)) 

# ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray) 
# ax2.set_title('Histogram of Oriented Gradients')

# plt.show()