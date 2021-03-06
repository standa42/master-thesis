import cv2

class TrackingPrediction:

    def __init__(self, classification, xmin, xmax, car_number = None, pneu_number = None):
        self.classification = classification
        self.xmin = xmin
        self.xmax = xmax
        self.car_number = car_number
        self.pneu_number = pneu_number

        self.xcenter = int(self.xmin + ((self.xmax - self.xmin) / 2))
        self.x_interval_size = self.xmax - self.xmin

    def inpaint_prediction(self, frame):
        # prediction interval line
        y = 1000
        if self.classification == "pneu":
            y = 900

        cv2.line(frame, (self.xmin, y), (self.xmax, y), (0, 0, 255), thickness=5)

        # prediction text
        text = f"car {self.car_number}"
        translate_back = 60
        if self.pneu_number is not None:
            text = text + f", pneu {self.pneu_number}"
            translate_back = 160


        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (self.xcenter - translate_back, y - 20)
        fontScale              = 1.5
        fontColor              = (0,0,255)
        thickness              = 3
        lineType               = 2

        cv2.putText(frame,
            text,
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

