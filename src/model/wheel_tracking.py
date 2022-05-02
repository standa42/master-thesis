import cv2
from numpy import full
from torch import _foreach_abs
from config.Config import Config
from operator import attrgetter
from src.model.bounding_box import BoundingBox

from src.model.tracking_prediction_v2 import TrackingPredictionV2

class WheelTracking:
    
    def __init__(self, camera, car_number, wheel_number, initial_bbox):
        self.camera = camera
        self.car_number = car_number
        self.wheel_number = wheel_number
        self.initial_bbox = initial_bbox
        self.current_bbox = initial_bbox
        self.tracking_failed_counter = 0

    def get_wheel_predictions(self):
        prediction = TrackingPredictionV2(self.camera, "pneu", self.current_bbox.xmin, self.current_bbox.xmax, self.car_number, self.wheel_number)
        return [prediction]

    def update_wheel_predictions(self, wheel_bboxes):
        if not wheel_bboxes:
            return self.tracking_fail()

        # find bbox with highest overlap
        overlap_sorted_bboxes = sorted(wheel_bboxes, key=lambda x: x.get_iou(self.current_bbox), reverse=True)
        # if there is no bbox with sufficient overlap - get strike
        bbox_highest_overlap = overlap_sorted_bboxes[0]
        if bbox_highest_overlap.get_iou(self.current_bbox) < 0.05:
            return self.tracking_fail()

        self.current_bbox = bbox_highest_overlap
        self.tracking_failed_counter = 0

        if bbox_highest_overlap.xmin > (1920/4):
            return self.tracking_fail()

        return True 

    def tracking_fail(self):
        # returns True if tracking is still ok, False otherwise
        self.tracking_failed_counter = self.tracking_failed_counter + 1
        return self.tracking_failed_counter < 3
