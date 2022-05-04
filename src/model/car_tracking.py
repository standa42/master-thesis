import cv2
from numpy import full
from torch import _foreach_abs
from config.Config import Config
from operator import attrgetter
from src.model.bounding_box import BoundingBox

from src.model.tracking_prediction_v2 import TrackingPredictionV2
from src.model.wheel_tracking import WheelTracking

class CarTracking:
    
    def __init__(self, car_number, initial_bbox, full_tracking = True):
        self.car_number = car_number
        self.initial_bbox = initial_bbox
        self.full_tracking = full_tracking
        self.current_bbox = initial_bbox
        self.tracking_failed_counter = 0

        self.first_wheel_a_complete = False 
        self.first_wheel_b_complete = False
        self.wheel_tracking_a = None
        self.wheel_tracking_b = None

        self.wheel_tracking_a_previous = None
        self.wheel_tracking_b_previous = None

    def get_car_predictions(self):
        camera_a_prediction = TrackingPredictionV2("a", "Car", self.current_bbox.xmin, self.current_bbox.xmax, self.car_number)
        camera_b_prediction = TrackingPredictionV2("b", "Car", self.current_bbox.xmin, self.current_bbox.xmax, self.car_number)
        return [camera_a_prediction, camera_b_prediction]

    def update_car_predictions(self, car_bboxes):
        if not car_bboxes:
            return self.tracking_fail()

        # find bbox with highest overlap
        overlap_sorted_bboxes = sorted(car_bboxes, key=lambda x: x.get_iou(self.current_bbox), reverse=True)
        # if there is no bbox with sufficient overlap - get strike
        bbox_highest_overlap = overlap_sorted_bboxes[0]
        if bbox_highest_overlap.get_iou(self.current_bbox) < 0.2:
            return self.tracking_fail()

        self.current_bbox = bbox_highest_overlap
        self.tracking_failed_counter = 0

        if bbox_highest_overlap.xmin > (1920/2.5):
            self.tracking_fail()
            return self.tracking_fail()

        return True 

    def tracking_fail(self):
        # returns True if tracking is still ok, False otherwise
        self.tracking_failed_counter = self.tracking_failed_counter + 1
        return self.tracking_failed_counter < 3

    def update_wheel_predictions(self, wheel_bboxes_a, wheel_bboxes_b):
        if not self.full_tracking:
            return 
        self.predictions = []

        if self.wheel_tracking_a is not None:
            tracking_ok = self.wheel_tracking_a.update_wheel_predictions(wheel_bboxes_a)

            if tracking_ok:
                self.predictions.extend(self.wheel_tracking_a.get_wheel_predictions())
            else:
                self.wheel_tracking_a_previous = self.wheel_tracking_a
                self.wheel_tracking_a = None
                self.first_wheel_a_complete = True
        else:
            if wheel_bboxes_a:
                leftmost_bbox = wheel_bboxes_a[0]

                self.wheel_tracking_a = WheelTracking("a", self.car_number, 3 if self.first_wheel_a_complete else 1, leftmost_bbox)
                self.predictions.extend(self.wheel_tracking_a.get_wheel_predictions())

        if self.wheel_tracking_b is not None:
            tracking_ok = self.wheel_tracking_b.update_wheel_predictions(wheel_bboxes_b)

            if tracking_ok:
                self.predictions.extend(self.wheel_tracking_b.get_wheel_predictions())
            else:
                self.wheel_tracking_b_previous = self.wheel_tracking_b
                self.wheel_tracking_b = None
                self.first_wheel_b_complete = True
        else:
            if wheel_bboxes_b:
                leftmost_bbox = wheel_bboxes_b[0]

                self.wheel_tracking_b = WheelTracking("b", self.car_number, 4 if self.first_wheel_a_complete else 2, leftmost_bbox)
                self.predictions.extend(self.wheel_tracking_b.get_wheel_predictions())

    def get_wheel_predictions(self):
        if not self.full_tracking:
            return []
        return self.predictions

    def get_class_and_sizes_predictions(self):
        predictions = []
        if self.wheel_tracking_a is not None:
            predictions.append(self.wheel_tracking_a.get_class_and_size_prediction())
        if self.wheel_tracking_b is not None:
            predictions.append(self.wheel_tracking_b.get_class_and_size_prediction())
        if self.wheel_tracking_a_previous is not None:
            predictions.append(self.wheel_tracking_a_previous.get_class_and_size_prediction())
        if self.wheel_tracking_b_previous is not None:
            predictions.append(self.wheel_tracking_b_previous.get_class_and_size_prediction())
        return predictions