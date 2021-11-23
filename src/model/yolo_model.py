import torch
import cv2

from config.Config import Config
from src.helpers.helper_functions import *

from src.model.bounding_box import BoundingBox


class YoloModel():
    """Supports loading of yolo models and basic operations on them"""

    def __init__(self, model):
        path = ''

        if model == 'tracking':
            path = Config.DataPaths.TrackingYoloModel
            self.inference_size = 640 

        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=path)

    def get_frame_with_bounding_boxes(self, frame):
        """Paints infered bounding boxes into frame"""
        bounding_boxes = self.get_bounding_boxes(frame)
        color = (255,0,0)
        for bounding_box in bounding_boxes:
            cv2.rectangle(frame, bounding_box.xmin, bounding_box.ymin, bounding_box.xmax, bounding_box.ymax, color, 10)
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