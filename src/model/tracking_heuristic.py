import cv2
from config.Config import Config
from operator import attrgetter

from src.model.tracking_prediction import TrackingPrediction

class TrackingHeuristic():
    # TODO: implement non-maximum suppresion like thing on x intervals

    def __init__(self):
        self.reset()

    def reset(self):
        self.max_index = -1
        self.bounding_boxes_history_camera_a = [None]*600
        self.bounding_boxes_history_camera_b = [None]*600
        self.predictions_history_camera_a = [None]*600
        self.predictions_history_camera_b = [None]*600

    def add_frame(self, frame_index, bounding_boxes, camera):
        if camera == "a":
            self.bounding_boxes_history_camera_a[frame_index] = bounding_boxes
        elif camera == "b":
            self.bounding_boxes_history_camera_b[frame_index] = bounding_boxes

    def get_prediction(self, frame_index, camera):
        if self.predictions_history_camera_a[frame_index] is not None:
            return self._get_prediction_for_camera(frame_index, camera)
        self._create_prediction(frame_index)
        return self._get_prediction_for_camera(frame_index, camera)

    def _get_prediction_for_camera(self, frame_index, camera):
        if camera == "a":
            return self.predictions_history_camera_a[frame_index]
        elif camera == "b":
            return self.predictions_history_camera_b[frame_index]

    def _create_prediction(self, frame_index):
        if frame_index > self.max_index:
            self.max_index = frame_index
        
        if frame_index == 0:
            self._create_first_prediction(frame_index)
        elif frame_index > 0:
            self._create_following_prediction(frame_index)

    def _create_first_prediction(self, frame_index):
        # get data for current frame
        bboxes_a = self.bounding_boxes_history_camera_a[frame_index]
        bboxes_b = self.bounding_boxes_history_camera_b[frame_index]

        # convert to predictions objects
        preds_a = [TrackingPrediction(bbox.classification, bbox.xmin, bbox.xmax) for bbox in bboxes_a]
        preds_b = [TrackingPrediction(bbox.classification, bbox.xmin, bbox.xmax) for bbox in bboxes_b]
        
        # split predictions to car and pneu
        preds_a_car = [pred for pred in preds_a if pred.classification == "car"]
        preds_a_pneu = [pred for pred in preds_a if pred.classification == "pneu"]
        preds_b_car = [pred for pred in preds_b if pred.classification == "car"]
        preds_b_pneu = [pred for pred in preds_b if pred.classification == "pneu"]
        
       # filter out near border bboxes    
        preds_a_car = self._filter_out_near_border(preds_a_car) 
        preds_a_pneu = self._filter_out_near_border(preds_a_pneu)
        preds_b_car = self._filter_out_near_border(preds_b_car) 
        preds_b_pneu = self._filter_out_near_border(preds_b_pneu)

        # duplicate supression
        preds_a_car = self._supress_duplicates(preds_a_car) 
        preds_a_pneu = self._supress_duplicates(preds_a_pneu)
        preds_b_car = self._supress_duplicates(preds_b_car) 
        preds_b_pneu = self._supress_duplicates(preds_b_pneu)
        
        # order them from right to left
        preds_a_car = self._order_bounding_boxes(preds_a_car)
        preds_a_pneu = self._order_bounding_boxes(preds_a_pneu)
        preds_b_car = self._order_bounding_boxes(preds_b_car)
        preds_b_pneu = self._order_bounding_boxes(preds_b_pneu)
        
        # try to match with previous predictions
        
        # try to match with second camera

        preds_a_car.extend(preds_a_pneu)
        preds_b_car.extend(preds_b_pneu)
    
        self.predictions_history_camera_a[frame_index] = preds_a_car
        self.predictions_history_camera_b[frame_index] = preds_b_car


    def _create_following_prediction(self, frame_index):
        # get data for current frame
        bboxes_a = self.bounding_boxes_history_camera_a[frame_index]
        bboxes_b = self.bounding_boxes_history_camera_b[frame_index]

        # convert to predictions objects
        preds_a = [TrackingPrediction(bbox.classification, bbox.xmin, bbox.xmax) for bbox in bboxes_a]
        preds_b = [TrackingPrediction(bbox.classification, bbox.xmin, bbox.xmax) for bbox in bboxes_b]
        
        # split predictions to car and pneu
        preds_a_car = [pred for pred in preds_a if pred.classification == "car"]
        preds_a_pneu = [pred for pred in preds_a if pred.classification == "pneu"]
        preds_b_car = [pred for pred in preds_b if pred.classification == "car"]
        preds_b_pneu = [pred for pred in preds_b if pred.classification == "pneu"]
        
        # filter out near border bboxes    
        preds_a_car = self._filter_out_near_border(preds_a_car) 
        preds_a_pneu = self._filter_out_near_border(preds_a_pneu)
        preds_b_car = self._filter_out_near_border(preds_b_car) 
        preds_b_pneu = self._filter_out_near_border(preds_b_pneu)

        # duplicate supression
        preds_a_car = self._supress_duplicates(preds_a_car) 
        preds_a_pneu = self._supress_duplicates(preds_a_pneu)
        preds_b_car = self._supress_duplicates(preds_b_car) 
        preds_b_pneu = self._supress_duplicates(preds_b_pneu)
        
        # order them from right to left
        preds_a_car = self._order_bounding_boxes(preds_a_car)
        preds_a_pneu = self._order_bounding_boxes(preds_a_pneu)
        preds_b_car = self._order_bounding_boxes(preds_b_car)
        preds_b_pneu = self._order_bounding_boxes(preds_b_pneu)
        
        # try to match with previous predictions
        
        # try to match with second camera

        for pred in preds_a_car:
            pred.car_number = 1
        for pred in preds_b_car:
            pred.car_number = 1
        for pred in preds_a_pneu:
            pred.car_number = 1
            pred.pneu_number = 1
        for pred in preds_b_pneu:
            pred.car_number = 1
            pred.pneu_number = 1

        preds_a_car.extend(preds_a_pneu)
        preds_b_car.extend(preds_b_pneu)
    
        self.predictions_history_camera_a[frame_index] = preds_a_car
        self.predictions_history_camera_b[frame_index] = preds_b_car

    def _supress_duplicates(self, preds):
        preds_result = []
        for pred in preds:
            preds_without_pred = set(preds)
            preds_without_pred.remove(pred)
            preds_without_pred = list(preds_without_pred)

            skip = False
            for pred2 in preds_without_pred:
                if self._supress_duplicates_remove_pred(pred, pred2):
                    skip = True

            if skip:
                continue 
            preds_result.append(pred)                       
            
        return preds_result

    def _supress_duplicates_remove_pred(self, pred1, pred2):
        """Returns true if pred1 should be removed on the base of pred2"""
        if self._calculate_overlap_part_from_pred1(pred1, pred2) > 0.2:
            if pred2.x_interval_size > pred1.x_interval_size:
                return True
        return False

    def _calculate_overlap_part_from_pred1(self, pred1, pred2):
        overlap = self._overlap_in_pixels(pred1, pred2)
        return overlap / float(pred1.x_interval_size)

    def _overlap_in_pixels(self, pred1, pred2): 
        if pred2.xmax < pred1.xmin:
            return 0
        elif pred2.xmin > pred1.xmax:
            return 0
        elif pred2.xmin <= pred1.xmin and pred2.xmax >= pred1.xmax:
            return pred1.x_interval_size
        elif pred2.xmin >= pred1.xmin and pred2.xmax <= pred1.xmax:
            return pred2.x_interval_size
        elif pred2.xmin <= pred1.xmin and pred2.xmax >= pred1.xmin:
            return pred2.xmax - pred1.xmin
        elif pred2.xmax >= pred1.xmax and pred2.xmin <= pred1.xmax:
            return pred1.xmax - pred2.xmin

    def _filter_out_near_border(self, preds):
        result_list = []
        xlimit = 350

        for pred in preds:
            if pred.xmin <= 0 + xlimit and pred.xmax <= 0 + xlimit:
                continue
            if pred.xmin >= 1920 - xlimit and pred.xmax >= 1920 - xlimit:
                continue
            result_list.append(pred)

        return result_list

    def _order_bounding_boxes(self, preds):
        preds.sort(key=attrgetter('xcenter'), reverse=True)        
        return preds
