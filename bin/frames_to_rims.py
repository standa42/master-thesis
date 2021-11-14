from kivy.app import App
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.clock import Clock
from src.framesloading.data_loader import frames_loader

import cv2
import numpy as np
from kivy.graphics.texture import Texture

from kivy.config import Config
Config.set('graphics', 'resizable', False)

import torch
import pandas as pd

# slider - on_touch_up, on_value


kv = '''
BoxLayout:
    orientation: 'horizontal'
    BoxLayout:
        size_hint: (1, 1)
        orientation: 'vertical'
        Image:
            id: upper_image
            size_hint: (1, 0.35)
            allow_stretch: True
            source: 'C:/Users/rnsk/Desktop/master-thesis-implementation/rims/data/frames/2019_05_13/10_48_35/A/frame100.jpg'
        BoxLayout:
            size_hint: (1, 0.15)
            orientation: 'horizontal'
        Image:
            id: lower_image
            size_hint: (1, 0.35)
            allow_stretch: True
            source: 'C:/Users/rnsk/Desktop/master-thesis-implementation/rims/data/frames/2019_05_13/10_48_35/A/frame100.jpg'
        BoxLayout:
            size_hint: (1, 0.15)
            orientation: 'horizontal'

    BoxLayout:
        size_hint: (None, 1)
        width: 200
        orientation: 'vertical'


        Label:
            id: frame_description
            text: 'text'
        Label:
            id: image_mode_description
            text: 'text'

        Button:
            text: 'downscaled'
            on_press:
                app.change_image_mode("downscaled")
        Button:
            text: 'gray'
            on_press:
                app.change_image_mode("gray")
        Button:
            text: 'blur'
            on_press:
                app.change_image_mode("blur")
        Button:
            text: 'canny'
            on_press:
                app.change_image_mode("canny")
        Button:
            text: 'result'
            on_press:
                app.change_image_mode("result")

        Label:
            id: slider_downscale_label
            text: 'Downscale'
        Slider:
            id: slider_downscale
            min: 1
            max: 100
            value: 40
            step: 1
            on_value: 
                app.downscale()

        Label:
            id: slider_blur_label
            text: 'Blur'
        Slider:
            id: slider_blur
            min: 1
            max: 10
            value: 2
            step: 1
            on_value: 
                app.blur()
        
        Label:
            id: slider_canny_1_label
            text: 'Canny 1 parameter'
        Label:
            id: slider_canny_2_label
            text: 'Canny 2 parameter'
        Slider:
            id: slider_canny_1
            min: 1
            max: 1000
            value: 150
            step: 1
            on_value: 
                app.canny()
        Slider:
            id: slider_canny_2
            min: 1
            max: 1000
            value: 250
            step: 1
            on_value: 
                app.canny()


'''


class FramesToRims(App):
    def build(self):
        return Builder.load_string(kv)

    # workaround to window resizing bug: https://github.com/kivy/kivy/issues/5359 part 1
    def to_window(self, x, y, initial=True, relative=False):
        return x,y

    def __init__(self, **kwargs):
        super(FramesToRims, self).__init__(**kwargs)
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)
        
        self.image_mode = "result"

        self.blur_value = 5
        self.canny_1 = 150
        self.canny_2 = 250
        self.downscale_value = 40

        # workaround to window resizing bug: https://github.com/kivy/kivy/issues/5359 part 2
        self.y = 0

        self.frames_a = []
        self.frames_b = []
        self.frames = frames_loader()
        self.frames.load_dataset()
        for a in range(0, len(self.frames.dataset.days)):
            for b in range(0, len(self.frames.dataset.days[a].ten_min_data)):
                for c in range(0, len(self.frames.dataset.days[a].ten_min_data[b].cameras)):
                    for d in range(0, len(self.frames.dataset.days[a].ten_min_data[b].cameras[c].frames)):
                        frame = self.frames.dataset.days[a].ten_min_data[b].cameras[c].frames[d]
                        if(c == 0):
                            self.frames_a.append(frame)
                        else:
                            self.frames_b.append(frame)
        self.frame_index = 0

        import os
        self.ds_frames = []

        fold = 'C:/Users/rnsk/Desktop/master-thesis-implementation/rims/data/yolo_dataset/first_part'
        for frame_file in os.listdir(fold): 
            self.ds_frames.append (os.path.join(fold, f"{frame_file}"))


        from PIL import Image
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/rnsk/Desktop/roboflow/yolov5/runs/train/exp5/weights/best.pt')

        pneu_counter = 0
        car_counter = 0

        for f in self.ds_frames:
            f_img = self.load_img(f) # f.path

            results = self.model(f_img, size=640)
         
            for index, row in results.pandas().xyxy[0].iterrows(): # img predictions
                crop_img = f_img[int(row['ymin']):int(row['ymax']), int(row['xmin']):int(row['xmax'])]
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                if row['name'] == 'pneu':
                    # crop_img = Image.fromarray(crop_img)
                    # crop_img.save('C:/Users/rnsk/Desktop/master-thesis-implementation/rims/data/yolo_object_cutout/pneu_from_1000_dataset//' + f'pneu_{str(pneu_counter).zfill(9)}.png')
                    # pneu_counter = pneu_counter + 1
                    pass
                elif row['name'] == 'car':
                    crop_img = Image.fromarray(crop_img)
                    crop_img.save('C:/Users/rnsk/Desktop/master-thesis-implementation/rims/data/test_del_me/' + f'car_{str(car_counter).zfill(9)}.png')
                    car_counter = car_counter + 1
                    pass
        quit()

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'left' or keycode[1] == 'a':
            self.frame_index = self.frame_index - 1
            self.reload_frame()
        elif keycode[1] == 'right' or keycode[1] == 'd':
            self.frame_index = self.frame_index + 1
            self.reload_frame()
        elif keycode[1] == 'q':
            self.frame_index_update = -1 
            Clock.unschedule(self.update)
            Clock.schedule_interval(self.update, 1 / 7.)
        elif keycode[1] == 'e':
            self.frame_index_update = 1
            Clock.unschedule(self.update)
            Clock.schedule_interval(self.update, 1 / 7.)
        elif keycode[1] == 'w':
            Clock.unschedule(self.update)
        elif keycode[1] == 's':
            self.next_mode()
        return True

    def update(self, dt):
        self.frame_index = self.frame_index + self.frame_index_update
        self.reload_frame()
    
    def reload_frame(self):
        self.root.ids.upper_image.texture = self.img_to_texture(self.add_circles_to_img(self.load_img(self.frames_a[self.frame_index].path)))
        self.root.ids.lower_image.texture = self.img_to_texture(self.add_circles_to_img(self.load_img(self.frames_b[self.frame_index].path)))

        self.root.ids.frame_description.text = "Frame description: " + str(self.frames_a[self.frame_index].number)
        
    def img_to_texture(self, img):
        buf1 = cv2.flip(img, 0)
        buf = buf1.tostring()
        texture1 = Texture.create(size=(img.shape[1], img.shape[0])) # colorfmt='bgr'
        if (self.image_mode == "canny" or self.image_mode == "blur" or self.image_mode == "gray"):
            texture1.blit_buffer(buf, bufferfmt='ubyte', colorfmt='luminance') # colorfmt='bgr'
        else:
            texture1.blit_buffer(buf, bufferfmt='ubyte', colorfmt='bgr') # colorfmt='bgr'
        # display image from the texture
        return texture1

    def load_img(self, path):
        img = cv2.imread(path) 
        return img 


    def add_circles_to_img(self, img):
        scale_percent = self.downscale_value # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        if(self.image_mode == "downscaled"):
            return resized

        # Convert to gray-scale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        if(self.image_mode == "gray"):
            return gray
        # Blur the image to reduce noise
        # img_blur = cv2.medianBlur(gray, 5)
        blur= cv2.GaussianBlur(gray, (self.blur_value, self.blur_value), cv2.BORDER_DEFAULT)
        if(self.image_mode == "blur"):
            return blur

        canny = cv2.Canny(blur, self.canny_1, self.canny_2)
        if(self.image_mode == "canny"):
            return canny
        # Apply hough transform on the image
        circles = cv2.HoughCircles(canny, cv2.HOUGH_GRADIENT, 1, img.shape[0]/64, param1=50, param2=35, minRadius=int(200 * scale_percent/100), maxRadius=int(300 * scale_percent/100)) # 50, 35, 235, 300
        # circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, img.shape[0]/64, param1=50, param2=35, minRadius=int(200 * scale_percent/100), maxRadius=int(300 * scale_percent/100)) # 50, 35, 235, 300
        # Draw detected circles
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Draw outer circle
                cv2.circle(resized, (i[0], i[1]), i[2], (0, 255, 0), 2) # center coor, radius, color, thickness
                # Draw inner circle
                cv2.circle(resized, (i[0], i[1]), 2, (0, 0, 255), 3)

        RGB_img = resized

        results = self.model(img, size=640)
         
        for index, row in results.pandas().xyxy[0].iterrows(): # img predictions
            clr = (255,0,0)
            if row['name'] == 'pneu':
                clr = (0,255,0)
            cv2.rectangle(img, (int(row['xmin']), int(row['ymin'])), (int(row['xmax']), int(row['ymax'])), clr, 10)

        return img

        return RGB_img

        

        # if circles is not None:
        #     counter = 0
        #     for circle in circles[0, :]:
        #         counter = counter + 1
        #         print(img.shape)
        #         print(circle)
        #         rectangle_size = 280
        #         rectangle_size_halved = int(rectangle_size/2)
        #         crop_img = final_resized[max(circle[1] - rectangle_size_halved,0):circle[1]+rectangle_size_halved, max(circle[0]- rectangle_size_halved,0):circle[0]+rectangle_size_halved]
        #         # crop_img = img[circle[1]:circle[1]+min(100, img.shape[0] - circle[1] -1), circle[0]:circle[0]+min(100, img.shape[1] - circle[0] -1)]
        #         cv2.imwrite(f"./data/3_extracted_rims/{str(a).zfill(3)}_{str(b).zfill(3)}_{str(c).zfill(3)}_{str(counter).zfill(2)}.jpg", crop_img)

    def change_image_mode(self, mode):
        self.image_mode = mode
        self.reload_frame()
        self.root.ids.image_mode_description.text = self.image_mode

    def next_mode(self):
        if self.image_mode == "result":
            self.image_mode = "downscaled"
        elif self.image_mode == "downscaled":
            self.image_mode = "gray"
        elif self.image_mode == "gray":
            self.image_mode = "blur"
        elif self.image_mode == "blur":
            self.image_mode = "canny"
        elif self.image_mode == "canny":
            self.image_mode = "result"
        else:
            self.image_mode = "result"
        self.reload_frame()
        self.root.ids.image_mode_description.text = self.image_mode

    def blur(self):
        blur_value = self.root.ids.slider_blur.value * 2 - 1
        self.root.ids.slider_blur_label.text = "Blur is: " + str(blur_value)
        self.blur_value = blur_value
        self.reload_frame()

    def canny(self):
        canny_1_value = self.root.ids.slider_canny_1.value
        canny_2_value = self.root.ids.slider_canny_2.value
        self.root.ids.slider_canny_1_label.text = "Canny 1 param: " + str(canny_1_value)
        self.canny_1 = canny_1_value
        self.root.ids.slider_canny_2_label.text = "Canny 2 param: " + str(canny_2_value)
        self.canny_2 = canny_2_value
        self.reload_frame()

    def downscale(self):
        downscale_value = self.root.ids.slider_downscale.value 
        self.root.ids.slider_downscale_label.text = "Downscale is: " + str(downscale_value)
        self.downscale_value = downscale_value
        self.reload_frame()


FramesToRims().run()