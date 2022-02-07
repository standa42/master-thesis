import cv2
import numpy as np

import kivy
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.screenmanager import Screen
from kivy.graphics.texture import Texture

from src.data.tracking_dataset.tracking_dataset_random import Tracking_dataset_random
from config.Config import Config

class HoughParametersScreen(Screen):
    
    # Initializations
    def __init__(self, **kwargs):
        super(HoughParametersScreen, self).__init__(**kwargs)

    def on_pre_enter(self):
        # Keyboard on_down callback
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)

        # set default values
        self.frame_index = 0
        self.frame_index_update = 0
        self.image_mode = "hough"
        self.clock_update_interval = 1.

        # loading algorithms parameters from config
        self.downscale_value = Config.Tracking.HoughTuningDownscaleValue
        self.blur_value = Config.Tracking.HoughTuningBlurValue
        self.canny_1_value = Config.Tracking.HoughTuningCanny1Value
        self.canny_2_value = Config.Tracking.HoughTuningCanny2Value
        self.hough_dp_value = Config.Tracking.HoughTuningDpValue
        self.hough_mindist_value = Config.Tracking.HoughTuningMinDistValue
        self.hough_param1_value = Config.Tracking.HoughTuningParam1Value
        self.hough_param2_value = Config.Tracking.HoughTuningParam2Value
        self.hough_minradius_value = Config.Tracking.HoughTuningMinRadiusValue
        self.hough_maxradius_value = Config.Tracking.HoughTuningMaxRadiusValue

        # init sliders with with algorithm parameters
        # self.init_sliders()

        # load dataset
        self.dataset = Tracking_dataset_random()

        # loads frames and updates labels information
        self.reload_frame()

    def init_sliders(self):
        # downscale
        self.ids.downscale_slider_slider.value  = self.downscale_value
        # blue
        self.ids.blur_slider_slider.value = int((self.blur_value - 1) / 2)
        # canny
        # self.ids.canny_slider_1_slider.value = self.canny_1_value
        # self.ids.canny_slider_2_slider.value = self.canny_2_value
        # hough
        self.ids.hough_dp_slider.value = self.hough_dp_value
        self.ids.hough_mindist_slider.value = self.hough_mindist_value
        self.ids.hough_param1_slider.value = self.hough_param1_value
        self.ids.hough_param2_slider.value = self.hough_param2_value
        self.ids.hough_minradius_slider.value = self.hough_minradius_value
        self.ids.hough_maxradius_slider.value = self.hough_maxradius_value

    # Updates
    def reload_frame(self):
        frame_index, frame = self.dataset.get_current()
        modified_frame = self.modify_image_accoring_to_image_mode(frame)
        texture = self.img_to_texture(modified_frame)
        self.ids.frame_image.texture = texture

        self.update_labels_values()
    
    def update_labels_values(self):
        # frame description
        self.ids.frame_description_label.text = "Frame: " + str(self.frame_index)
        # downscale
        self.ids.downscale_slider_label.text = "Downscale is: " + str(self.downscale_value)
        # blur
        self.ids.blur_slider_label.text = "Blur is: " + str(self.blur_value)
        # canny
        # self.ids.canny_slider_1_label.text = "Canny 1 param: " + str(self.canny_1_value)
        # self.ids.canny_slider_2_label.text = "Canny 2 param: " + str(self.canny_2_value)
        # hough
        self.ids.hough_dp_label.text = "Dp: " + str(self.hough_dp_value)
        self.ids.hough_mindist_label.text = "MinDist: " + str(self.hough_mindist_value)
        self.ids.hough_param1_label.text = "Param1: " + str(self.hough_param1_value)
        self.ids.hough_param2_label.text = "Param2: " + str(self.hough_param2_value)
        self.ids.hough_minradius_label.text = "MinRadius: " + str(self.hough_minradius_value)
        self.ids.hough_maxradius_label.text = "MaxRadius: " + str(self.hough_maxradius_value)
        
    def modify_image_accoring_to_image_mode(self, original_frame):
        # downscaling
        downscale_to_percent_of_original = self.downscale_value # percent of original size
        
        downscaled_width = int(original_frame.shape[1] * downscale_to_percent_of_original / 100)
        downscaled_height = int(original_frame.shape[0] * downscale_to_percent_of_original / 100)
        
        resized_frame = cv2.resize(original_frame, (downscaled_width, downscaled_height), interpolation = cv2.INTER_CUBIC)
        
        if(self.image_mode == "downscaled"):
            return resized_frame

        # converting to gray-scale
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        if(self.image_mode == "gray"):
            return gray_frame

        # histogram normalization
        gray_frame = cv2.equalizeHist(gray_frame)

        # blurring the image (to reduce noise)
        blur_frame = cv2.GaussianBlur(gray_frame, (self.blur_value, self.blur_value), cv2.BORDER_DEFAULT)
        if(self.image_mode == "blur"):
            return blur_frame

        # canny edge filter
        canny_frame = blur_frame
        # canny_frame = cv2.Canny(blur_frame, self.canny_1_value, self.canny_2_value)
        # if(self.image_mode == "canny"):
        #     return canny_frame
        
        # hough transform
        circles = cv2.HoughCircles(canny_frame,
                    cv2.HOUGH_GRADIENT,
                    self.hough_dp_value, # 1, # dp (size of accumulator)
                    self.hough_mindist_value, # 75, # minDist
                    param1 = self.hough_param1_value, # param1=50, # gradient for edge detection
                    param2 = self.hough_param2_value, # param2=35, # accumulator threshold
                    minRadius = self.hough_minradius_value, # minRadius=70, # minRadius, originally int(180 * downscale_to_percent_of_original/100)
                    maxRadius = self.hough_maxradius_value # maxRadius=130 # maxRadius, originally int(330 * downscale_to_percent_of_original/100)
                    )

        # draw detected circles
        hough_frame = resized_frame
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Draw outer circle
                cv2.circle(hough_frame, (i[0], i[1]), i[2], (0, 255, 0), 2) # center coor, radius, color, thickness
                # Draw inner circle
                cv2.circle(hough_frame, (i[0], i[1]), 2, (0, 0, 255), 3)

        if(self.image_mode == "hough"):
            return hough_frame
    
    def img_to_texture(self, img):
        buffer1 = cv2.flip(img, 0)
        buffer2 = buffer1.tostring()
        texture = Texture.create(size=(img.shape[1], img.shape[0])) # colorfmt='bgr'
        if (self.image_mode == "canny" or self.image_mode == "blur" or self.image_mode == "gray"):
            texture.blit_buffer(buffer2, bufferfmt='ubyte', colorfmt='luminance') # colorfmt='bgr'
        else:
            texture.blit_buffer(buffer2, bufferfmt='ubyte', colorfmt='bgr') # colorfmt='bgr'
        return texture

    # image mode change
    def change_image_mode(self, mode):
        self.image_mode = mode
        self.reload_frame()

    # sliders control
    def downscale(self):
        self.downscale_value = self.ids.downscale_slider_slider.value 
        self.reload_frame()

    def blur(self):
        self.blur_value = self.ids.blur_slider_slider.value * 2 - 1
        self.reload_frame()

    def canny(self):
        # self.canny_1_value = self.ids.canny_slider_1_slider.value
        # self.canny_2_value = self.ids.canny_slider_2_slider.value
        self.reload_frame()

    def hough(self):
        self.hough_dp_value = self.ids.hough_dp_slider.value
        self.hough_mindist_value = self.ids.hough_mindist_slider.value
        self.hough_param1_value = self.ids.hough_param1_slider.value
        self.hough_param2_value = self.ids.hough_param2_slider.value
        self.hough_minradius_value = self.ids.hough_minradius_slider.value
        self.hough_maxradius_value = self.ids.hough_maxradius_slider.value
        self.reload_frame()

    # frame movement control
    def next_frame(self):
        frame_index, frame = self.dataset.get_next()
        if frame_index == None:
            return
        self.frame_index = frame_index
        self.reload_frame()

    def previous_frame(self):
        frame_index, frame = self.dataset.get_previous()
        if frame_index == None:
            return
        self.frame_index = frame_index
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
        self.frame_index = self.frame_index + self.frame_index_update
        if self.frame_index_update == 1:
            _, _ = self.dataset.get_next()
        elif self.frame_index_update == -1:
            _, _ = self.dataset.get_previous()
        self.reload_frame()
    
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
        elif keycode[1] == 's':
            self.next_mode()
        return True

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    # next mode selector
    def next_mode(self):
        if self.image_mode == "hough":
            self.image_mode = "downscaled"
        elif self.image_mode == "downscaled":
            self.image_mode = "gray"
        elif self.image_mode == "gray":
            self.image_mode = "blur"
        elif self.image_mode == "blur":
        #     self.image_mode = "canny"
        # elif self.image_mode == "canny":
            self.image_mode = "hough"
        else:
            self.image_mode = "hough"
        self.reload_frame()