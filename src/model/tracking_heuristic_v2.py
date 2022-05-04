import cv2
from torch import _foreach_abs
from config.Config import Config
from operator import attrgetter
from src.model.bounding_box import BoundingBox

from src.model.car_tracking import CarTracking

class TrackingHeuristicV2:

    def __init__(self):
        self.reset()

    def reset(self):
        self.tracked_car = None
        self.car_counter = 0
        self.highest_predict_frame_index = -1
        
        self.train_data = [None] * 600
        self.predictions = [None] * 600
        self.class_and_sizes_predictions = [None] * 600
        pass

    def add_tracking_data(self, frame_index, bounding_boxes, camera):
        # remove bboxes near sides
        bounding_boxes = self.remove_bboxes_near_sides(bounding_boxes)
        # duplicate suppression in individual cathegories
        bounding_boxes = self.duplicate_suppression(bounding_boxes)
        # sort from left to right
        bounding_boxes = self.sort_from_leftmost(bounding_boxes)
        # add to data 
        if camera == "a":
            self.train_data[frame_index] = (bounding_boxes, None)
        elif camera == "b":
            self.train_data[frame_index] = (self.train_data[frame_index][0], bounding_boxes)

    def impaint_predictions(self, frame_index, frame_a, frame_b):
        if self.highest_predict_frame_index == -1:
            # init frame
            self.init_frame()
        elif frame_index < self.highest_predict_frame_index:
            # return some already made prediction
            pass 
        else: 
            # do prediction
            self.predict(frame_index)

        # set highest reached predict frame index
        if frame_index > self.highest_predict_frame_index:
            self.highest_predict_frame_index = frame_index

        # impaint prediction to given frame index and return
        return self.impaint_prediction(frame_index, frame_a, frame_b)

    def init_frame(self):
        # do prediction for initial frame
        # get car predictions from camera a
        car_bboxes = list(filter(lambda x: x.classification == 'car', self.train_data[0][0])) 
        # setup tracking
        if car_bboxes:
            full_tracking = True 
            leftmost_bbox = car_bboxes[0]

            if leftmost_bbox.xmax > (1920/3): # NOTE: here was /2
                full_tracking = False

            self.create_car(leftmost_bbox, full_tracking)
            self.predictions[0] = self.tracked_car.get_car_predictions()

    def predict(self, frame_index):
        # do prediciton for general frame
        # get car predictions from camera a
        car_bboxes = list(filter(lambda x: x.classification == 'car', self.train_data[frame_index][0])) 
        wheel_bboxes_a = list(filter(lambda x: x.classification == 'pneu', self.train_data[frame_index][0])) 
        wheel_bboxes_b = list(filter(lambda x: x.classification == 'pneu', self.train_data[frame_index][1])) 
        if self.tracked_car is not None:
            tracking_ok = self.tracked_car.update_car_predictions(car_bboxes)

            if tracking_ok:
                self.predictions[frame_index] = self.tracked_car.get_car_predictions()
                self.tracked_car.update_wheel_predictions(wheel_bboxes_a, wheel_bboxes_b)
                self.predictions[frame_index].extend(self.tracked_car.get_wheel_predictions())
                self.class_and_sizes_predictions[frame_index] = self.tracked_car.get_class_and_sizes_predictions()
            else:
                self.tracked_car = None
        else: 
            if car_bboxes:
                full_tracking = True 
                leftmost_bbox = car_bboxes[0]

                if leftmost_bbox.xmax > (1920/2):
                    full_tracking = False

                self.create_car(leftmost_bbox, full_tracking)
                self.predictions[frame_index] = self.tracked_car.get_car_predictions()

    def create_car(self, bbox, full_tracking):
        self.car_counter = self.car_counter + 1
        self.tracked_car = CarTracking(self.car_counter, bbox, full_tracking)

    def impaint_prediction(self, frame_index, frame_a, frame_b):
        # impaint prediction to frames based on prediction generated for given frame_index
        if self.predictions[frame_index] is not None:
            for prediction in self.predictions[frame_index]:
                if prediction.camera == "a":
                    prediction.inpaint_prediction(frame_a)
                elif prediction.camera == "b":
                    prediction.inpaint_prediction(frame_b)
        return (frame_a, frame_b)

    def remove_bboxes_near_sides(self, bounding_boxes, percent_of_image_to_cut_left = 0.2, percent_of_image_to_cut_right = 0.3):
        # remove bboxes near sides and return modified collection
        pixels_of_image_to_cut_left = 1920 * percent_of_image_to_cut_left
        pixels_of_image_to_cut_right = 1920 * percent_of_image_to_cut_right
        new_bounding_boxes = []
        for bbox in bounding_boxes:
            if not (bbox.xmin > (1920-pixels_of_image_to_cut_right) or bbox.xmax < pixels_of_image_to_cut_left):
                new_bounding_boxes.append(bbox)
        return new_bounding_boxes

    def duplicate_suppression(self, bounding_boxes):
        # remove bboxes that has high overlap and return modified collection
        new_bboxes = []
        
        for bbox in bounding_boxes:
            bounding_boxes_without = bounding_boxes[:]

            should_remain = True
            for another_bbox in bounding_boxes_without:
                if bbox.classification == another_bbox.classification and bbox.get_iou(another_bbox) > 0.3 and bbox.get_area() < another_bbox.get_area():
                    # TODO: in condition handle case where areas are equal - now it will keep both
                    should_remain = False

            if should_remain:
                new_bboxes.append(bbox)
        
        return new_bboxes
    
    def sort_from_leftmost(self, boudning_boxes):
        return sorted(boudning_boxes, key=lambda x: x.xmin)