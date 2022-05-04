import kivy
from kivy.app import App
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.uix.screenmanager import Screen
from kivy.graphics.texture import Texture

import cv2
import torch
import numpy as np
import pandas as pd

from bin.trackingapp.MenuScreen import MenuScreen
from bin.trackingapp.HoughParametersScreen import HoughParametersScreen
from bin.trackingapp.DetectionVisualisationScreen import DetectionVisualisationScreen
from bin.trackingapp.SizeEstimationTestingScreen import SizeEstimationTestingScreen

class TrackingApp(App):
    # workaround to window resizing bug: https://github.com/kivy/kivy/issues/5359 part 1
    def to_window(self, x, y, initial=True, relative=False):
        return x,y

    def __init__(self, **kwargs):
        super(TrackingApp, self).__init__(**kwargs)

        # workaround to window resizing bug: https://github.com/kivy/kivy/issues/5359 part 2
        self.y = 0

        # resize window
        Window.size = (1500, 800)


if __name__ == '__main__':
    kivy.Config.set('graphics', 'resizable', True)
    TrackingApp().run()
