import cv2
import numpy as np

import kivy
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.uix.screenmanager import Screen
from kivy.graphics.texture import Texture
from src.model.yolo_model import YoloModel

from src.data.screws_segmentation_dataset.screws_segmentation_dataset import BoltsEstimationDataset
from config.Config import Config

from skimage.measure import EllipseModel

class SizeEstimationTestingScreen(Screen):
    
    # Initializations
    def __init__(self, **kwargs):
        super(SizeEstimationTestingScreen, self).__init__(**kwargs)

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
        self.dataset = BoltsEstimationDataset()

        self.yolo_model_bolts = YoloModel('size_estimation_256')

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
        working_image = original_frame.copy()

        # get bounding boxes bolts + rim
        all_bounding_boxes = self.yolo_model_bolts.get_bounding_boxes(working_image)
        rim_bbox = list(filter(lambda x: x.classification == 'Rim', all_bounding_boxes))
        bolts_bboxs = list(filter(lambda x: x.classification == 'Bolt', all_bounding_boxes))

        # impaint bolts bboxes
        for bounding_box2 in bolts_bboxs:
            cv2.rectangle(working_image, (bounding_box2.xmin, bounding_box2.ymin), (bounding_box2.xmax, bounding_box2.ymax), (0,0,255), 2)

        # imprint pcd ellipse
        centers_wheel_bboxes = []
        for bounding_box3 in bolts_bboxs:
            bbox_center = bounding_box3.get_center()
            centers_wheel_bboxes.append( (float(bbox_center[0]), float(bbox_center[1]))      )
        bolts_axes = (None, None)
        if len(centers_wheel_bboxes) == 5:
            rim_ellipse_points =np.array([np.array(xi) for xi in centers_wheel_bboxes]) #np.array(centers_wheel_bboxes)
            x = rim_ellipse_points[:, 0]
            y = rim_ellipse_points[:, 1]
            ell = EllipseModel()
            done = ell.estimate(rim_ellipse_points)
            if done:
                xc, yc, a, b, theta = ell.params
                center_coordinates = (int(xc), int(yc))
                axesLength = (int(a), int(b))
                bolts_axes = (a,b)
                angle = int(theta*180/np.pi)
                startAngle = 0
                endAngle = 360
                color = (0, 255, 0)
                thickness = 2
                working_image = cv2.ellipse(working_image, center_coordinates, axesLength, angle,
                                        startAngle, endAngle, color, thickness)

        # crop image to the rim
        working_image_rim_crop = working_image.copy()
        if rim_bbox:
            rim_bbox = rim_bbox[0]
            padding = 5
            xmin = max(rim_bbox.xmin - padding, 0)
            xmax = min(rim_bbox.xmax + padding, 770)
            ymin = max(rim_bbox.ymin - padding, 0)
            ymax = min(rim_bbox.ymax + padding, 770)
            
            xsize = xmax - xmin
            ysize = ymax - ymin

            working_image_rim_crop = working_image[ymin:ymin+ysize, xmin:xmin+xsize]

        # otsu thresholding
        gray_frame = cv2.cvtColor(working_image_rim_crop, cv2.COLOR_BGR2GRAY)
        blur_frame = cv2.GaussianBlur(gray_frame, (self.blur_value, self.blur_value), cv2.BORDER_DEFAULT)
        ret3,otsu = cv2.threshold(blur_frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        self.image_mode = "canny"

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

        ell = EllipseModel()
        done = ell.estimate(rim_ellipse_points)

        rim_axes = (None, None)

        if done:
            xc, yc, a, b, theta = ell.params

            center_coordinates = (int(xc), int(yc))
            axesLength = (int(a), int(b))
            rim_axes = (a,b)
            angle = int(theta*180/np.pi)
            startAngle = 0
            endAngle = 360
            color = (0, 255, 0)
            thickness = 2
            otsu_in_rgb = cv2.ellipse(otsu_in_rgb, center_coordinates, axesLength, angle,
                                    startAngle, endAngle, color, thickness)

        for point in rim_ellipse_points_in_drawing_format:
            otsu_in_rgb = cv2.circle(otsu_in_rgb, (int(point[1]), int(point[0])), radius=0, color=(0, 0, 255), thickness=3)

        # update UI with axes of ellipses and estimation of rim size in inches
        self.ids.textx.text = f"{bolts_axes}"
        self.ids.textxx.text = f"{rim_axes}"

        pcd_size = 112.0 #mm
        inch_to_mm = 25.4 # multiply

        if rim_axes[0] is not None and bolts_axes[0] is not None:
            ratio = max(rim_axes) / max(bolts_axes)
            self.ids.textxxx.text = f"{(pcd_size * ratio) / inch_to_mm}"

        return otsu_in_rgb

    
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