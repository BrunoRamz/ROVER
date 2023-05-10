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

    def get_frame(self, cap, scaling_factor):
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

scaling_factor = float(os.getenv("SCALING_FACTOR"))
cap = cv2.VideoCapture(0)
previous_frame = frame.get_frame(cap, scaling_factor)
current_frame = frame.get_frame(cap, scaling_factor)
next_frame = frame.get_frame(cap, scaling_factor)

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
    next_frame = frame.get_frame(cap, scaling_factor)

    key = cv2.waitKey(10)
    if key == 27:
        break

cv2.destroyAllWindows()
