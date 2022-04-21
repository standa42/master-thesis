import cv2
# from keras.saving.saved_model.load import load
import numpy as np
import torch
import PIL
from PIL import Image
import requests
import json
import time

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
from src.model.tracking_prediction import TrackingPrediction

from skimage.measure import EllipseModel

import numpy as np
from numpy.linalg import eig, inv
def fitEllipse(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a

def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])


def ellipse_angle_of_rotation( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return 0.5*np.arctan(2*b/(a-c))


def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])

def ellipse_angle_of_rotation2( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    if b == 0:
        if a > c:
            return 0
        else:
            return np.pi/2
    else: 
        if a > c:
            return np.arctan(2*b/(a-c))/2
        else:
            return np.pi/2 + np.arctan(2*b/(a-c))/2

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
        self.clock_update_interval = .3

        self.gold_enabled = True
        self.yolo_enabled = True
        self.hough_enabled = True

        self.upper_detection_boxes = []
        self.lower_detection_boxes = []

        self.video_dataset = Video_dataset()
        self.video_dataset_pairs = self.video_dataset.get_all_video_pairs()
        self.video_dataset_current_pair_frames_paths = None if self.frame_index == -1 else self.video_dataset_pairs[self.video_index].generate_paths_to_frames()

        self.yolo_model = YoloModel('tracking')
        self.yolo_model_bolts = YoloModel('wheel_bolts_detection')

        self.tracking_heuristic = TrackingHeuristic()

        self.reload_frame()

    # Updates
    def reload_frame(self):
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
            
        # inpaint tracking data
        frame_a = self.inpaint_tracking(frame_a, "a")
        frame_b = self.inpaint_tracking(frame_b, "b")
        # update frames in scene
        self.ids.upper_frame_image.texture = self.img_to_texture(frame_a)
        self.ids.lower_frame_image.texture = self.img_to_texture(frame_b)
        # udpate labels
        self.update_labels_values()
    
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
        # inpaint bounding boxes
        color = (255,0,0)
        thickness = 7
        bounding_boxes = self.yolo_model.get_bounding_boxes(frame)
        for bounding_box in bounding_boxes:
            cv2.rectangle(frame, (bounding_box.xmin, bounding_box.ymin), (bounding_box.xmax, bounding_box.ymax), color, thickness)
        # add data to tracking heuristic
        self.tracking_heuristic.add_frame(self.frame_index, bounding_boxes, camera)

        # display cropped pneus
        for bbox in [b for b in bounding_boxes if b.classification == "pneu"]:
            bbox.make_centered_wheel_bounding_box()
            cropped_image = bbox.get_crop_from_image(original_frame)

            cropped_image = np.uint8(cropped_image)
            origo = np.copy(cropped_image)
            origo = cv2.cvtColor(origo, cv2.COLOR_RGB2BGR)
            # cropped_image = self.hough_on_screws(cropped_image)

            print("wheel bolts yolo detection")
            wheel_bolts_bounding_boxes = self.yolo_model_bolts.get_bounding_boxes(cropped_image)
            for bounding_box2 in wheel_bolts_bounding_boxes:
                cv2.rectangle(cropped_image, (bounding_box2.xmin, bounding_box2.ymin), (bounding_box2.xmax, bounding_box2.ymax), color, thickness)

            # elipse inprint
            centers_wheel_bboxes = []
            for bounding_box2 in wheel_bolts_bounding_boxes:
                bbox_center = bounding_box2.get_center()
                centers_wheel_bboxes.append( (float(bbox_center[0]), float(bbox_center[1]))      )
            
            if len(centers_wheel_bboxes) == 5:

                a_points =np.array([np.array(xi) for xi in centers_wheel_bboxes]) #np.array(centers_wheel_bboxes)
                x = a_points[:, 0]
                y = a_points[:, 1]

                ell = EllipseModel()
                done = ell.estimate(a_points)

                if done:
                    xc, yc, a, b, theta = ell.params

                    center_coordinates = (int(xc), int(yc))
                    axesLength = (int(a), int(b))
                    angle = int(theta*180/np.pi)
                    startAngle = 0
                    endAngle = 360
                    color = (0, 255, 0)
                    thickness = 2
                    cropped_image = cv2.ellipse(cropped_image, center_coordinates, axesLength, angle,
                                            startAngle, endAngle, color, thickness)


                # a = fitEllipse(x,y)
                # center = ellipse_center(a)
                # #phi = ellipse_angle_of_rotation(a)
                # phi = ellipse_angle_of_rotation2(a)
                # axes = ellipse_axis_length(a)

                # center_coordinates = (int(center[0]),int(center[1]))
                # axesLength = (int(axes[0]), int(axes[1]))
                # angle = 30
                # startAngle = 0
                # endAngle = 360
                # color = (0, 255, 0)
                # thickness = 2
                # cropped_image = cv2.ellipse(cropped_image, center_coordinates, axesLength, angle,
                #                         startAngle, endAngle, color, thickness)

                

            # elipse inprint end Ellipse((xc, yc), 2*a, 2*b, theta*180/np.pi, edgecolor='red', facecolor='none')

            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)

            texture = self.img_to_texture(cropped_image)

            box = BoxLayout()
            box.orientation = "horizontal"

            image1 = Image()
            image1.size_hint = (0.5, 1)
            image1.allow_stretch = True
            image1.texture = texture

            image2 = Image()
            image2.size_hint = (0.5, 1)
            image2.allow_stretch = True

            ################################################
            downscaled_cropped_image = cv2.resize(origo, (256, 256), interpolation=cv2.INTER_CUBIC)
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
                image2.texture = self.img_to_texture(loaded_image)
            except:
                pass
            ################################################


            if camera == "a":
                self.upper_detection_boxes.append(box)
                self.ids.upper_box.add_widget(box)
            elif camera == "b":
                self.lower_detection_boxes.append(box)
                self.ids.lower_box.add_widget(box)

            box.add_widget(image1)
            box.add_widget(image2)

            

        # end
        return frame

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
