import cv2
# from keras.saving.saved_model.load import load
import numpy as np
import torch
import PIL
from PIL import Image
import requests
import json
import time
import copy

import kivy
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.screenmanager import Screen
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image

from config.Config import Config
from src.data.video.video_dataset import Video_dataset
from src.model.yolo_model import YoloModel
from src.model.tracking_heuristic import TrackingHeuristic
from src.model.tracking_heuristic_v2 import TrackingHeuristicV2
from src.model.tracking_prediction import TrackingPrediction

from skimage.measure import EllipseModel

import numpy as np
from numpy.linalg import eig, inv

class DetectionVisualisationScreen(Screen):

    # Initializations
    def __init__(self, **kwargs):
        super(DetectionVisualisationScreen, self).__init__(**kwargs)

    def on_pre_enter(self):
        # Keyboard on_down callback
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)
        
        # set default values
        self.video_index = 0
        self.frame_index = 0 #TODO: make it -1 for labelled dataset after I have completed that dataset loading
        self.frame_index_update = 0
        self.clock_update_interval = .35

        self.gold_enabled = True
        self.yolo_enabled = True
        self.hough_enabled = True

        self.upper_detection_boxes = []
        self.lower_detection_boxes = []

        self.video_dataset = Video_dataset()
        self.video_dataset_pairs = self.video_dataset.get_all_video_pairs()
        self.video_dataset_current_pair_frames_paths = None if self.frame_index == -1 else self.video_dataset_pairs[self.video_index].generate_paths_to_frames()

        self.yolo_model = YoloModel('tracking')
        self.yolo_model_bolts = YoloModel('size_estimation_256')

        self.tracking_heuristic = TrackingHeuristicV2()

        self.reload_frame()

    # Updates
    def reload_frame(self):
        self.reset_ui_before_reload_frame()

        start_time = time.time()
        frame_a, frame_b = self.get_frames()
        original_frame_a = frame_a.copy()
        original_frame_b = frame_b.copy()
        original_frame_b = cv2.flip(original_frame_b, 1)
        # flip frame b
        frame_b = cv2.flip(frame_b, 1)
        # print hough
        if self.hough_enabled:
            frame_a = self.hough(frame_a)
            frame_b = self.hough(frame_b)
        # print yolo
        if self.yolo_enabled:
            for db in self.upper_detection_boxes:
                self.ids.upper_box.remove_widget(db)
            self.upper_detection_boxes = []
            frame_a = self.yolo(frame_a, "a", original_frame_a)

            for db in self.lower_detection_boxes:
                self.ids.lower_box.remove_widget(db)
            self.lower_detection_boxes = []
            frame_b = self.yolo(frame_b, "b", original_frame_b)
            
        # TODO: inpaint tracking data
        # frame_a = self.inpaint_tracking(self.frame_index, frame_a, "a")
        # frame_b = self.inpaint_tracking(self.frame_index, frame_b, "b")
        frame_a, frame_b = self.tracking_heuristic.impaint_predictions(self.frame_index, frame_a, frame_b)
        self.update_wheel_hystersis()

        end_time = time.time()
        self.ids.pipeline_time_label.text = f'Pipeline time[s]: {"{:.2f}".format(end_time-start_time)}'

        # update frames in scene
        self.ids.upper_frame_image.texture = self.img_to_texture(frame_a)
        self.ids.lower_frame_image.texture = self.img_to_texture(frame_b)
        # udpate labels
        self.update_labels_values()

    def update_wheel_hystersis(self):
        if self.tracking_heuristic.tracked_car is None:
            self.ids.upper_box_data_pneu_class_one.text = ""
            self.ids.upper_box_data_pneu_size_one.text = ""
            self.ids.upper_box_data_pneu_class_two.text = ""
            self.ids.upper_box_data_pneu_size_two.text = ""
            self.ids.upper_box_data_pneu_class_three.text = ""
            self.ids.upper_box_data_pneu_size_three.text = ""
            self.ids.upper_box_data_pneu_class_four.text = ""
            self.ids.upper_box_data_pneu_size_four.text = ""

        class_and_sizes_predictions = self.tracking_heuristic.class_and_sizes_predictions[self.frame_index]
        if class_and_sizes_predictions is not None:
            pneu_1 = list(filter(lambda x: x.pneu_id == '1', class_and_sizes_predictions))
            if pneu_1:
                pneu_1 = pneu_1[0]
                self.ids.upper_box_data_pneu_class_one.text = pneu_1.pneu_class
                self.ids.upper_box_data_pneu_size_one.text = pneu_1.pneu_avg_size
            pneu_2 = list(filter(lambda x: x.pneu_id == '2', class_and_sizes_predictions))
            if pneu_2:
                pneu_2 = pneu_2[0]
                self.ids.upper_box_data_pneu_class_two.text = pneu_2.pneu_class
                self.ids.upper_box_data_pneu_size_two.text = pneu_2.pneu_avg_size
            pneu_3 = list(filter(lambda x: x.pneu_id == '3', class_and_sizes_predictions))
            if pneu_3:
                pneu_3 = pneu_3[0]
                self.ids.upper_box_data_pneu_class_three.text = pneu_3.pneu_class
                self.ids.upper_box_data_pneu_size_three.text = pneu_3.pneu_avg_size
            pneu_4 = list(filter(lambda x: x.pneu_id == '4', class_and_sizes_predictions))
            if pneu_4:
                pneu_4 = pneu_4[0]
                self.ids.upper_box_data_pneu_class_four.text = pneu_4.pneu_class
                self.ids.upper_box_data_pneu_size_four.text = pneu_4.pneu_avg_size

    def reset_ui_before_reload_frame(self):
        blank_image = np.zeros((256,256,3), np.uint8)
        blank_image = self.img_to_texture(blank_image)
        self.ids.upper_box_images_top.texture = blank_image
        self.ids.upper_box_images_bottom.texture = blank_image
        self.ids.lower_box_images_top.texture = blank_image
        self.ids.lower_box_images_bottom.texture = blank_image
        self.ids.upper_box_images_middle.text = f""
        self.ids.lower_box_images_middle.text = f""

        self.ids.upper_box_data_pcd_ellipse_axes.text = ""
        self.ids.upper_box_data_rim_ellipse_axes.text = ""
        self.ids.upper_box_data_ratio_of_main_axes.text = ""
        self.ids.upper_box_data_rim_estimation_inch.text = ""
        self.ids.lower_box_data_pcd_ellipse_axes.text = ""
        self.ids.lower_box_data_rim_ellipse_axes.text = ""
        self.ids.lower_box_data_ratio_of_main_axes.text = ""
        self.ids.lower_box_data_rim_estimation_inch.text = ""
    
    def update_labels_values(self):
        # video description
        if self.video_index >= 0:
            self.ids.video_description_label.text = "Video: " + str(self.video_index) + "\n" + self.video_dataset_pairs[self.video_index].video_a.datetime
        elif self.video_index == -1:
            self.ids.video_description_label.text = "Video: " + str("labelled dataset")
        # frame description
        self.ids.frame_description_label.text = "Frame: " + str(self.frame_index)
    
    def img_to_texture(self, img):
        buffer1 = cv2.flip(img, 0)
        buffer2 = buffer1.tostring()
        texture = Texture.create(size=(img.shape[1], img.shape[0]))
        texture.blit_buffer(buffer2, bufferfmt='ubyte', colorfmt='bgr') 
        return texture

    def get_frames(self):
        if self.video_index >= 0:
            frame_a_path, frame_b_path =  self.video_dataset_current_pair_frames_paths[self.frame_index]
            frame_a = cv2.imread(frame_a_path)
            frame_b = cv2.imread(frame_b_path)
            return (frame_a, frame_b)
        elif self.video_index == -1:
            pass # TODO: implement for second dataset
            # get frame based on frame_index
            # inpaint bounding boxes
            # return

    def hough(self, frame):
        frame_copy = frame.copy()
        downscale_to_percent_of_original = Config.Tracking.HoughTuningDownscaleValue
        downscaled_width = int(frame_copy.shape[1] * downscale_to_percent_of_original / 100)
        downscaled_height = int(frame_copy.shape[0] * downscale_to_percent_of_original / 100)
        resized_frame = cv2.resize(frame_copy, (downscaled_width, downscaled_height), interpolation = cv2.INTER_CUBIC)
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.equalizeHist(gray_frame)
        blur_frame = cv2.GaussianBlur(gray_frame, (Config.Tracking.HoughTuningBlurValue, Config.Tracking.HoughTuningBlurValue), cv2.BORDER_DEFAULT)
        canny_frame = blur_frame
        circles = cv2.HoughCircles(canny_frame,
                    cv2.HOUGH_GRADIENT,
                    Config.Tracking.HoughTuningDpValue, 
                    Config.Tracking.HoughTuningMinDistValue, 
                    param1 = Config.Tracking.HoughTuningParam1Value, 
                    param2 = Config.Tracking.HoughTuningParam2Value, 
                    minRadius = Config.Tracking.HoughTuningMinRadiusValue,
                    maxRadius = Config.Tracking.HoughTuningMaxRadiusValue 
                    )
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Draw outer circle
                cv2.circle(frame,
                    (int(i[0] * (100.0/downscale_to_percent_of_original)), int(i[1] * (100.0/downscale_to_percent_of_original))),
                    int(i[2] * (100.0/downscale_to_percent_of_original)), 
                    (0, 255, 0), 
                    7) # center coor, radius, color, thickness
                # Draw inner circle
                cv2.circle(frame, 
                    (int(i[0] * (100.0/downscale_to_percent_of_original)), int(i[1] * (100.0/downscale_to_percent_of_original))),
                    2, 
                    (0, 255, 0), 
                    6)
        return frame

    def yolo(self, frame, camera, original_frame):
        # inpaint bounding boxes to the whole frame
        bounding_boxes = self.yolo_model.get_bounding_boxes(frame)
        for bounding_box in bounding_boxes:
            cv2.rectangle(frame, (bounding_box.xmin, bounding_box.ymin), (bounding_box.xmax, bounding_box.ymax), color=(255,0,0), thickness=7)

        # delete bounding boxes of wheels with too high aspect ratio
        bounding_boxes = list(filter(lambda bbox: bbox.classification != 'pneu' or bbox.is_aspect_ratio_lower_than(5/3.0), bounding_boxes))

        # pneu_bboxes
        pneu_bboxes = [b for b in bounding_boxes if b.classification == "pneu"]
        pneu_bboxes_for_heuristic = copy.deepcopy(bounding_boxes)

        # work with crops of the wheels
        for bbox_index in range(len(pneu_bboxes)):
            bbox = pneu_bboxes[bbox_index]

            # crop
            bbox.make_centered_wheel_bounding_box()
            cropped_image_to_modify = bbox.get_crop_from_image(original_frame)
            cropped_image_to_modify = np.uint8(cropped_image_to_modify)
            crop_original = np.copy(cropped_image_to_modify)
            crop_original = cv2.cvtColor(crop_original, cv2.COLOR_RGB2BGR)

            # classify cropped image
            label, classification_image_texture = self.class_inference_on_server(crop_original)

            # object detection on crop -> rim and bolts bboxes
            crop_detail_bounding_boxes = self.yolo_model_bolts.get_bounding_boxes(cropped_image_to_modify)
            # impaint bolts bboxes
            bolts_bboxes = list(filter(lambda x: x.classification == "Bolt", crop_detail_bounding_boxes))
            for bounding_box2 in bolts_bboxes:
                cv2.rectangle(cropped_image_to_modify, (bounding_box2.xmin, bounding_box2.ymin), (bounding_box2.xmax, bounding_box2.ymax), color=(255,0,0), thickness=7)
            # compute pcd ellipse if 5 bolts are present, impaint it and get ellipse axes
            if len(bolts_bboxes) == 5:
                centers_of_bolts_bboxes = []
                for bounding_box in bolts_bboxes:
                    bbox_center = bounding_box.get_center()
                    centers_of_bolts_bboxes.append( (float(bbox_center[0]), float(bbox_center[1])))
                points =np.array([np.array(xi) for xi in centers_of_bolts_bboxes]) 
                cropped_image_to_modify, pcd_axes = self.ellipse_fit(cropped_image_to_modify, points)
            else:
                pcd_axes = (None, None)

            # compute rim ellipse and impaint it into image
            # but first check, whether rim bbox exists and whether it has some distance from the edges
            rim_bboxes = list(filter(lambda x: x.classification == "Rim", crop_detail_bounding_boxes))
            rim_axes = (None, None)
            rim_diameter_inches = None
            if rim_bboxes:
                rim_bbox = rim_bboxes[0]
                if rim_bbox.xmin > 10 and rim_bbox.xmax < 760:
                    cropped_image_to_modify, rim_axes = self.rim_ellipse_estimation(cropped_image_to_modify, rim_bbox)
                    rim_diameter_inches = self.compute_rim_diameter(pcd_axes, rim_axes)

                    # prepare image with impainted ellipses for UI
                    cropped_image_to_modify = cv2.cvtColor(cropped_image_to_modify, cv2.COLOR_RGB2BGR)
                    image1_texture = self.img_to_texture(cropped_image_to_modify)

            # update UI
            if camera == "a":
                self.ids.upper_box_images_bottom.texture = classification_image_texture
                self.ids.upper_box_images_middle.text = f"Predicted class: {label}"
                if pcd_axes[0] is not None and rim_axes[0] is not None:
                    self.ids.upper_box_images_top.texture = image1_texture
                    self.ids.upper_box_data_pcd_ellipse_axes.text = f'PCD ellipse axes are: ({"{:.2f}".format(pcd_axes[0])}, {"{:.2f}".format(pcd_axes[1])})'
                    self.ids.upper_box_data_rim_ellipse_axes.text = f'Rim ellipse axes are: ({"{:.2f}".format(rim_axes[0])}, {"{:.2f}".format(rim_axes[1])})'
                    self.ids.upper_box_data_ratio_of_main_axes.text = f'Ratio of main axes is: {"{:.2f}".format(max(rim_axes) / float(max(pcd_axes)))}'
                    self.ids.upper_box_data_rim_estimation_inch.text = f'Rim diameter is: {"{:.2f}".format(rim_diameter_inches)} inches'
            elif camera == "b":
                self.ids.lower_box_images_bottom.texture = classification_image_texture
                self.ids.lower_box_images_middle.text = f"Predicted class: {label}"
                if pcd_axes[0] is not None and rim_axes[0] is not None:
                    self.ids.lower_box_images_top.texture = image1_texture
                    self.ids.lower_box_data_pcd_ellipse_axes.text = f'PCD ellipse axes are: ({"{:.2f}".format(pcd_axes[0])}, {"{:.2f}".format(pcd_axes[1])})'
                    self.ids.lower_box_data_rim_ellipse_axes.text = f'Rim ellipse axes are: ({"{:.2f}".format(rim_axes[0])}, {"{:.2f}".format(rim_axes[1])})'
                    self.ids.lower_box_data_ratio_of_main_axes.text = f'Ratio of main axes is: {"{:.2f}".format(max(rim_axes) / float(max(pcd_axes)))}'
                    self.ids.lower_box_data_rim_estimation_inch.text = f'Rim diameter is: {"{:.2f}".format(rim_diameter_inches)} inches'
                    
            pneu_bboxes_for_heuristic[bbox_index].pneu_class = label
            pneu_bboxes_for_heuristic[bbox_index].pneu_size = rim_diameter_inches

        # add data to tracking heurustic
        self.tracking_heuristic.add_tracking_data(self.frame_index, pneu_bboxes_for_heuristic, camera)

        # yolo end
        return frame

    def compute_rim_diameter(self, pcd_axes, rim_axes):
        pcd_size = 112.0 #mm
        inch_to_mm = 25.4 

        if rim_axes[0] is not None and pcd_axes[0] is not None:
            ratio = max(rim_axes) / max(pcd_axes)
            rim_size_in_inches = (pcd_size * ratio) / inch_to_mm
            return rim_size_in_inches
        else:
            return None

    def class_inference_on_server(self, image):
        """Sends image to classification server and return predicted label together with representative image for that label"""
        downscaled_cropped_image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
        downscaled_cropped_image = cv2.flip(downscaled_cropped_image, 1) # the whole frame is mirrored, so we need to correct that for classification 
        image_texture = None
        label = None
        try:
            addr = 'http://127.0.0.1:5000'
            test_url = addr + '/api/test'

            # prepare headers for http request
            content_type = 'image/jpeg'
            headers = {'content-type': content_type}

            # encode image as jpeg
            _, img_encoded = cv2.imencode('.jpg', downscaled_cropped_image)
            # send http request with image and receive response
            response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
            # decode response
            # print(json.loads(response.text))
            message_content = json.loads(response.text)["message"]
            loaded_image = cv2.imread(Config.DataPaths.UniqueRimsCollage + str(message_content) + ".png")
            label = str(message_content)
            if str(message_content) == "unrecognized":
                loaded_image = np.zeros((256,256,3), np.uint8)
                loaded_image = self.img_to_texture(loaded_image)
                image_texture = loaded_image
                label = "other"
            else:
                image_texture = self.img_to_texture(loaded_image)
        except:
            pass
        return label, image_texture

    def ellipse_fit(self, image, points):
        """Accepts image and sampled points, performs ellipse fitting and returns image with impatinted ellipse and ellipse axes"""
        ell = EllipseModel()
        done = ell.estimate(points)

        axes = None

        if done:
            xc, yc, a, b, theta = ell.params

            center_coordinates = (int(xc), int(yc))
            axesLength = (int(a), int(b))
            axes = (a,b)
            angle = int(theta*180/np.pi)
            startAngle = 0
            endAngle = 360
            color = (0, 255, 0)
            thickness = 3
            image = cv2.ellipse(image, center_coordinates, axesLength, angle,
                                    startAngle, endAngle, color, thickness)
        
        return image, axes

    def hough_on_screws(self, cropped_pneu):
        cropped_pneu_copy = cropped_pneu.copy()

        y_step = int(770/2.8)
        low_y = 0 + y_step
        high_y = 770 - y_step
        y_cropped_image = cropped_pneu_copy[low_y:high_y, :]

        gray_frame = cv2.cvtColor(y_cropped_image, cv2.COLOR_BGR2GRAY)
        
        blur_frame = cv2.GaussianBlur(gray_frame, (3, 3), cv2.BORDER_DEFAULT)

        canny_frame = blur_frame

        circles = cv2.HoughCircles(canny_frame,
                    cv2.HOUGH_GRADIENT,
                    1, 
                    35, 
                    param1 = 70, 
                    param2 = 22, 
                    minRadius = 1,
                    maxRadius = 20
                    )
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Draw outer circle
                cv2.circle(cropped_pneu,
                    (i[0], i[1]+low_y),
                    i[2], 
                    (0, 0, 255), 
                    7) # center coor, radius, color, thickness
                # Draw inner circle
                cv2.circle(cropped_pneu, 
                    (i[0], i[1]+low_y),
                    2, 
                    (0, 0, 255), 
                    6)
        return cropped_pneu

    def inpaint_tracking(self, frame, camera):
        predictions = self.tracking_heuristic.get_prediction(self.frame_index, camera)

        for pred in predictions:
            pred.inpaint_prediction(frame)

        return frame

    # video movement control
    def next_video(self):
        if self.video_index < (len(self.video_dataset_pairs) - 1):
            self.video_index = self.video_index + 1
            self.frame_index = 0
            self.tracking_heuristic.reset()
        self.video_dataset_current_pair_frames_paths = None if self.frame_index == -1 else self.video_dataset_pairs[self.video_index].generate_paths_to_frames()
        self.reload_frame()

    def previous_video(self):
        if self.video_index > 0: # TODO: -1 for labelled dataset
            self.video_index = self.video_index - 1
            self.frame_index = 0
            self.tracking_heuristic.reset()
        self.video_dataset_current_pair_frames_paths = None if self.frame_index == -1 else self.video_dataset_pairs[self.video_index].generate_paths_to_frames()
        self.reload_frame()

    # frame movement control
    def next_frame(self):
        if self.frame_index < 599:
            self.frame_index = self.frame_index + 1
        self.reload_frame()

    def previous_frame(self):
        if self.frame_index > 0:
            self.frame_index = self.frame_index - 1
        self.reload_frame()

    # animation control
    def animate_frames_forward(self):
        self.frame_index_update = 1
        Clock.unschedule(self.animate_update)
        Clock.schedule_interval(self.animate_update, self.clock_update_interval)
    
    def animate_frames_backward(self):
        self.frame_index_update = -1
        Clock.unschedule(self.animate_update)
        Clock.schedule_interval(self.animate_update, self.clock_update_interval)
    
    def animate_frames_pause(self):
        self.frame_index_update = 0
        Clock.unschedule(self.animate_update)
    
    def animate_update(self, dt):
        if self.frame_index_update == 1:
            self.next_frame()
        elif self.frame_index_update == -1:
            self.previous_frame()

    # toggle mode buttons
    # def toggle_gold(self):
    #     if self.gold_enabled:
    #         self.gold_enabled = False
    #         self.ids.toggle_gold_button.text = 'Enable Gold labels'
    #     else:
    #         self.gold_enabled = True
    #         self.ids.toggle_gold_button.text = 'Disable Gold labels'

    def toggle_yolo(self):
        if self.yolo_enabled:
            self.yolo_enabled = False
            self.ids.toggle_yolo_button.text = 'Enable YOLO'
        else:
            self.yolo_enabled = True
            self.ids.toggle_yolo_button.text = 'Disable YOLO'

    def toggle_hough(self):
        if self.hough_enabled:
            self.hough_enabled = False
            self.ids.toggle_hough_button.text = 'Enable Hough'
        else:
            self.hough_enabled = True
            self.ids.toggle_hough_button.text = 'Disable Hough'
    
    # keyboard callbacks
    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'left' or keycode[1] == 'a':
            self.previous_frame()
        elif keycode[1] == 'right' or keycode[1] == 'd':
            self.next_frame()
        elif keycode[1] == 'q':
            self.animate_frames_backward()
        elif keycode[1] == 'e':
            self.animate_frames_forward()
        elif keycode[1] == 'w':
            self.animate_frames_pause()
        elif keycode[1] == 'y':
            self.previous_video()
        elif keycode[1] == 'c':
            self.next_video()
        return True

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def rim_ellipse_estimation(self, image, rim_bbox):
        # get working copy of image
        working_image_rim_crop = image.copy()

        # crop image to the rim
        padding = 5
        xmin = max(rim_bbox.xmin - padding, 0)
        xmax = min(rim_bbox.xmax + padding, 770)
        ymin = max(rim_bbox.ymin - padding, 0)
        ymax = min(rim_bbox.ymax + padding, 770)
        
        xsize = xmax - xmin
        ysize = ymax - ymin

        working_image_rim_crop = working_image_rim_crop[ymin:ymin+ysize, xmin:xmin+xsize]

        # otsu thresholding
        gray_frame = cv2.cvtColor(working_image_rim_crop, cv2.COLOR_BGR2GRAY)
        blur_frame = cv2.GaussianBlur(gray_frame, (3, 3), cv2.BORDER_DEFAULT)
        ret3,otsu = cv2.threshold(blur_frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # raycasting
        otsu_height, otsu_width = otsu.shape
        otsu_height_half = otsu_height/2
        otsu_height_tenth = otsu_height/10

        x_rays = [otsu_height_half, otsu_height_half + otsu_height_tenth, otsu_height_half + otsu_height_tenth + otsu_height_tenth, otsu_height_half - otsu_height_tenth, otsu_height_half - otsu_height_tenth - otsu_height_tenth] 
        x_rays = list(map(lambda x: int(x), x_rays))
        
        otsu_width_half = otsu_width/2
        otsu_width_tenth = otsu_width/10

        y_rays = [otsu_width_half, otsu_width_half + otsu_width_tenth, otsu_width_half + otsu_width_tenth + otsu_width_tenth, otsu_width_half - otsu_width_tenth, otsu_width_half - otsu_width_tenth - otsu_width_tenth] 
        y_rays = list(map(lambda x: int(x), y_rays))

        rim_ellipse_points = []
        # from left
        for ray in x_rays:
            found = False
            for i in range(len(otsu[ray])):
                if otsu[ray][i] == 255:
                    found = True 
                    rim_ellipse_points.append([float(ray), float(i)])
                    break
            if found:
                continue
        
        # from right
        for ray in x_rays:
            found = False
            for i in range(len(otsu[ray])):
                if otsu[ray][len(otsu[ray]) - 1 - i] == 255:
                    found = True 
                    rim_ellipse_points.append([float(ray), float(len(otsu[ray]) - 1 - i)])
                    break
            if found:
                continue

        # from top
        for ray in y_rays:
            found = False
            for i in range(len(otsu)):
                if otsu[i][ray] == 255:
                    found = True 
                    rim_ellipse_points.append([float(i), float(ray)])
                    break
            if found:
                continue

        # from bottom
        for ray in y_rays:
            found = False
            for i in range(len(otsu)):
                if otsu[len(otsu) - 1 - i][ray] == 255:
                    found = True 
                    rim_ellipse_points.append([float(len(otsu) - 1 - i), float(ray)])
                    break
            if found:
                continue

        # convert otsu image to rgb
        otsu_in_rgb = working_image_rim_crop
        self.image_mode = "not-important"
        
        # convert points
        rim_ellipse_points_in_drawing_format = rim_ellipse_points 
        rim_ellipse_points = list(map(lambda x: (x[1],x[0]), rim_ellipse_points))
        rim_ellipse_points = np.array(rim_ellipse_points)

        # ellipse fit and draw
        otsu_in_rgb, rim_axes = self.ellipse_fit(otsu_in_rgb, rim_ellipse_points)

        # impaint individual points of raycast hit
        for point in rim_ellipse_points_in_drawing_format:
            otsu_in_rgb = cv2.circle(otsu_in_rgb, (int(point[1]), int(point[0])), radius=0, color=(0, 0, 255), thickness=3)

        return otsu_in_rgb, rim_axes

        