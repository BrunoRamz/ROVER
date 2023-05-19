import cv2
import numpy as np
import os


from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())

ESC_KEY = int(os.getenv("ESC_KEY"))
WAIT_KEY = int(os.getenv("WAIT_KEY"))
SCALING_FACTOR = float(os.getenv("SCALING_FACTOR"))
KERNEL_SIZE = int(os.getenv("KERNEL_SIZE"))
DEVICE_INDEX = int(os.getenv("DEVICE_INDEX"))


class ColorSpaces:

    def __init__(self):
        self.capture = cv2.VideoCapture(DEVICE_INDEX)

    def convert_hsv(self):
        hsv = cv2.cvtColor(self.resize_image_frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 70, 60])
        upper = np.array([50, 150, 255])
        mask = cv2.inRange(hsv, lower, upper)
        image_bitwise_and = cv2.bitwise_and(
            self.resize_image_frame, self.resize_image_frame, mask=mask
        )
        image_media_blurred = cv2.medianBlur(image_bitwise_and, KERNEL_SIZE)

        return image_media_blurred

    def start_tracking_color_spaces(self):
        while True:
            _, self.frame = self.capture.read()
            self.resize_image_frame = cv2.resize(
                self.frame, None, fx=SCALING_FACTOR,
                fy=SCALING_FACTOR, interpolation=cv2.INTER_AREA
            )
            self.hsv_convertion = self.convert_hsv()
            cv2.imshow('Output', self.hsv_convertion)
            cv2.imshow('Input', self.resize_image_frame)

            key = cv2.waitKey(WAIT_KEY)
            if key == ESC_KEY:
                break
        cv2.destroyAllWindows()


color_spaces = ColorSpaces()
color_spaces.start_tracking_color_spaces()
