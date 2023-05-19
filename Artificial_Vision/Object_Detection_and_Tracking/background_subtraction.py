import cv2
import os


from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())

ESC_KEY = int(os.getenv("ESC_KEY"))
WAIT_KEY = int(os.getenv("WAIT_KEY"))
SCALING_FACTOR = float(os.getenv("SCALING_FACTOR"))
DEVICE_INDEX = int(os.getenv("DEVICE_INDEX"))
HISTORY = int(os.getenv("HISTORY"))
RATE = float(os.getenv("RATE"))


class BackgroundSubtraction:
    def __init__(self):
        self.capture = cv2.VideoCapture(DEVICE_INDEX)
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.learning_rate = RATE/HISTORY

    def compute_maks(self, frame):
        mask = self.background_subtractor.apply(
            frame, learningRate=self.learning_rate
        )
        grayscale_to_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        return grayscale_to_rgb

    def star_background_subtraction(self):
        while True:
            _, self.frame = self.capture.read()
            self.resize_image_frame = cv2.resize(
                self.frame, None, fx=SCALING_FACTOR,
                fy=SCALING_FACTOR, interpolation=cv2.INTER_AREA
            )
            self.mask = self.compute_maks(self.resize_image_frame)
            cv2.imshow('Intput', self.resize_image_frame)
            cv2.imshow('Output', self.mask & self.resize_image_frame)

            key = cv2.waitKey(WAIT_KEY)
            if key == ESC_KEY:
                break

        self.capture.release()
        cv2.destroyAllWindows()


background_subtraction = BackgroundSubtraction()
background_subtraction.star_background_subtraction()
