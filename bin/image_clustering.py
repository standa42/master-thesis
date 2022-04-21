import gc
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


if __name__ == "__main__":

    safe_mkdir(Config.DataPaths.ScaledDownCropsFolder)
    crops_files = [Config.DataPaths.ScaledDownCropsFolder + f for f in listdir(Config.DataPaths.ScaledDownCropsFolder) if isfile(join(Config.DataPaths.ScaledDownCropsFolder, f))]

    crops_files = crops_files[:1000]

    crops = [cv2.imread(c) for c in crops_files]



    
    features = []
    for c in crops:
        hog_features = hog(c, multichannel=True)
        hog_features = [f*100.0 for f in hog_features]
        features.append(hog_features)
        
    features = PCA(n_components=8).fit_transform(features)

    labels = GaussianMixture(n_components=16).fit_predict(features)


    safe_mkdir_clean(Config.DataPaths.ClusterWheelsFolder)
    for i in range(len(labels)):
        label = labels[i]
        cluster_folder = Config.DataPaths.ClusterWheelsFolder + f"{label}/"
        safe_mkdir(cluster_folder)
        
        original_path = crops_files[i]
        filename = original_path.split('/')[-1]

        new_path = cluster_folder + filename
        copyfile(original_path, new_path)


