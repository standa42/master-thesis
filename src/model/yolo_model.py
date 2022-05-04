import torch
import cv2

from config.Config import Config
from src.helpers.helper_functions import *

from src.model.bounding_box import BoundingBox

import time

class YoloModel():
    """Supports loading of yolo models and basic operations on them"""

    def __init__(self, model):
        path = ''

        if model == 'tracking':
            path = Config.DataPaths.TrackingYoloModel
            self.inference_size = 640 
        elif model == 'wheel_bolts_detection':
            path = Config.DataPaths.WheelBoltsDetectionYoloModel
            self.inference_size = 768
        elif model == 'size_estimation_256':
            path = Config.DataPaths.SizeEstimation256Model
            self.inference_size = 256 

        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=path)

    def get_frame_with_bounding_boxes(self, frame, thickness=5, color=(255,0,0)):
        """Paints infered bounding boxes into frame"""
        bounding_boxes = self.get_bounding_boxes(frame)
        for bounding_box in bounding_boxes:
            cv2.rectangle(frame, (bounding_box.xmin, bounding_box.ymin), (bounding_box.xmax, bounding_box.ymax), color, thickness)
        return frame

    def get_bounding_boxes(self, frame):
        """Gets infered bounding boxes"""
        bounding_boxes = []

        for index, row in self._inference(frame).pandas().xyxy[0].iterrows():
            bounding_box = BoundingBox(row['name'], int(row['xmin']), int(row['xmax']), int(row['ymin']), int(row['ymax']) )
            bounding_boxes.append(bounding_box)

        return bounding_boxes
    
    def _inference(self, frame):
        """Runs inference on the model"""
        return self.model(frame, size=self.inference_size)
