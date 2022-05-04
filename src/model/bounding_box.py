import cv2
from PIL import Image
from statistics import mean

class BoundingBox:

    def __init__(self, classification, xmin, xmax, ymin, ymax):
        self.classification = classification
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.pneu_class = None
        self.pneu_size = None

    def get_crop_from_image(self, image):
        """Returns crop from the image the is specified by bounding box"""
        # crop
        cropped_image = image[self.ymin:self.ymax, self.xmin:self.xmax]
        # get correct color ordering
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        # convert back from numpy array to Image object
        cropped_image = Image.fromarray(cropped_image)
        return cropped_image

    def get_center(self):
        return (int((self.xmin + self.xmax)/2), int((self.ymin + self.ymax)/2))

    def get_area(self):
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)

    def get_intersection_area(self, another_bbox):
        overlap_area_xmin = max(self.xmin, another_bbox.xmin)
        overlap_area_ymin = max(self.ymin, another_bbox.ymin)
        overlap_area_xmax = min(self.xmax, another_bbox.xmax)
        overlap_area_ymax = min(self.ymax, another_bbox.ymax)

        # tests for non-overlapping dimensions
        if overlap_area_xmax <= overlap_area_xmin:
            return 0.0
        if overlap_area_ymax <= overlap_area_ymin:
            return 0.0

        x_size = overlap_area_xmax - overlap_area_xmin
        y_size = overlap_area_ymax - overlap_area_ymin

        return float(x_size * y_size) 
    
    def get_union_area(self, another_bbox):
        return self.get_area() + another_bbox.get_area() - self.get_intersection_area(another_bbox)

    def get_iou(self, another_bbox):
        return self.get_intersection_area(another_bbox) / self.get_union_area(another_bbox)

    def is_aspect_ratio_lower_than(self, ratio):
        x_size = self.xmax - self.xmin
        y_size = self.ymax - self.ymin
        bbox_ratio = max(x_size, y_size) / float(min(x_size, y_size))
        return bbox_ratio < ratio

    def make_centered_wheel_bounding_box(self):
        """Changes bounding box, so that center is in the original one and size is 770x770 fitted into original image of size 1920x1080"""
        center_x = int(mean([self.xmin, self.xmax]))
        center_y = int(mean([self.ymin, self.ymax]))

        size = 770
        half_size = size/2

        left = center_x - half_size
        right = center_x + half_size
        top = center_y - half_size
        bottom = center_y + half_size

        if left < 0:
            left = 0
            right = size
        if right >= 1920:
            right = 1920
            left = 1920 - size
        if top < 0:
            top = 0
            bottom = size
        if bottom >= 1080:
            bottom = 1080
            top = 1080 - size

        self.xmin = int(left)
        self.xmax = int(right)
        self.ymin = int(top)
        self.ymax = int(bottom)

