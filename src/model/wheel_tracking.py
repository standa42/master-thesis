import cv2
from numpy import full
from torch import _foreach_abs
from config.Config import Config
from operator import attrgetter
from src.model.bounding_box import BoundingBox
import statistics

from src.model.tracking_prediction_v2 import TrackingPredictionV2
from src.model.size_prediction import SizePrediction

class WheelTracking:
    
    def __init__(self, camera, car_number, wheel_number, initial_bbox):
        self.camera = camera
        self.car_number = car_number
        self.wheel_number = wheel_number
        self.initial_bbox = initial_bbox
        self.current_bbox = initial_bbox
        self.tracking_failed_counter = 0
        
        self.classifications_storage = []
        self.sizes_storage = []

    def get_wheel_predictions(self):
        prediction = TrackingPredictionV2(self.camera, "pneu", self.current_bbox.xmin, self.current_bbox.xmax, self.car_number, self.wheel_number)
        return [prediction]

    def get_class_and_size_prediction(self):
        most_common_class = "None" 
        most_common_class_count = 0
        classification_storage_without_none = list(filter(lambda x: x is not None, self.classifications_storage))
        if classification_storage_without_none:
            most_common_class = max(set(classification_storage_without_none), key=classification_storage_without_none.count)
            most_common_class_count = len(list(filter(lambda x: x == most_common_class, classification_storage_without_none)))
        classification_text = f"Wheel{self.wheel_number} class: '{most_common_class}' with {most_common_class_count} votes out of {len(classification_storage_without_none)}"

        average_size = "None"
        average_size_samples = 0
        size_storage_without_none = list(filter(lambda x: x is not None, self.sizes_storage))
        if size_storage_without_none:
            average_size = statistics.mean(size_storage_without_none)
            average_size_samples = len(size_storage_without_none)
        sizes_text = f'Wheel{self.wheel_number} avg size: {"{:.2f}".format(average_size) if average_size != "None" else "None"} out of {average_size_samples} samples'

        prediction = SizePrediction(camera=self.camera, pneu_id=f"{self.wheel_number}", pneu_class=classification_text, pneu_avg_size=sizes_text)
        return prediction

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

        self.classifications_storage.append(bbox_highest_overlap.pneu_class)
        self.sizes_storage.append(bbox_highest_overlap.pneu_size)

        if bbox_highest_overlap.xmin > (1920/4):
            return self.tracking_fail()

        return True 

    def tracking_fail(self):
        # returns True if tracking is still ok, False otherwise
        self.tracking_failed_counter = self.tracking_failed_counter + 1
        return self.tracking_failed_counter < 3
