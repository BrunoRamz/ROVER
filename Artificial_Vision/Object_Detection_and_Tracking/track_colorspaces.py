import cv2
import numpy as np
import os


from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())


class ColorSpaces:

    def get_frame(self, cap):
        scaling_factor = float(os.getenv("SCALING_FACTOR"))
        _, self.frame = cap.read()
        self.resize_image_frame = cv2.resize(
            self.frame, None, fx=scaling_factor,
            fy=scaling_factor, interpolation=cv2.INTER_AREA
        )

        return self.resize_image_frame

    def convert_hsv(self, frame):
        kernel_size = int(os.getenv("KERNEL_SIZE"))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 70, 60])
        upper = np.array([50, 150, 255])
        mask = cv2.inRange(hsv, lower, upper)
        image_bitwise_and = cv2.bitwise_and(frame, frame, mask=mask)
        image_media_blurred = cv2.medianBlur(image_bitwise_and, kernel_size)

        return image_media_blurred


esc_key = int(os.getenv("ESC_KEY"))
wait_key = int(os.getenv("WAIT_KEY"))
device_index = int(os.getenv("DEVICE_INDEX"))
color_spaces = ColorSpaces()
cap = cv2.VideoCapture(device_index)

while True:

    frame = color_spaces.get_frame(cap)
    hsv_convertion = color_spaces.convert_hsv(frame)

    cv2.imshow('Output', hsv_convertion)
    cv2.imshow('Input', frame)

    key = cv2.waitKey(wait_key)
    if key == esc_key:
        break

cv2.destroyAllWindows()
