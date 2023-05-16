import cv2
import os


from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())


class Frame:

    def frame_difference(
            self, previous_frame, current_frame, next_frame
    ):
        self.difference_current_next_frame = cv2.absdiff(
            next_frame, current_frame
        )
        self.difference_current_previous_frame = cv2.absdiff(
            current_frame, previous_frame
        )

        return cv2.bitwise_and(
            self.difference_current_next_frame,
            self.difference_current_previous_frame
            )

    def get_frame(self, cap):
        scaling_factor = float(os.getenv("SCALING_FACTOR"))
        _, self.frame = cap.read()
        self.resize_image_frame = cv2.resize(
            self.frame, None, fx=scaling_factor,
            fy=scaling_factor, interpolation=cv2.INTER_AREA
        )
        self.convert_grayscale = cv2.cvtColor(
            self.frame, cv2.COLOR_RGB2GRAY
        )

        return self.convert_grayscale


frame = Frame()

esc_key = int(os.getenv("ESC_KEY"))
wait_key = int(os.getenv("WAIT_KEY"))
device_index = int(os.getenv("DEVICE_INDEX"))
cap = cv2.VideoCapture(device_index)
previous_frame = frame.get_frame(cap)
current_frame = frame.get_frame(cap)
next_frame = frame.get_frame(cap)

while True:
    cv2.imshow(
        'Object Movement',
        frame.frame_difference(
            previous_frame,
            current_frame,
            next_frame)
    )
    previous_frame = current_frame
    current_frame = next_frame
    next_frame = frame.get_frame(cap)

    key = cv2.waitKey(wait_key)
    if key == esc_key:
        break

cv2.destroyAllWindows()
