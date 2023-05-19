import cv2
import os


from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())


class Frame:
    def __init__(self):
        self.capture = cv2.VideoCapture(int(os.getenv("DEVICE_INDEX")))
        self.previous_frame = self.get_frame()
        self.current_frame = self.get_frame()
        self.next_frame = self.get_frame()

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

    def get_frame(self):
        _, self.frame = self.capture.read()
        self.scaling_factor = float(os.getenv("SCALING_FACTOR"))
        self.resize_image = cv2.resize(
            self.frame, None, fx=self.scaling_factor,
            fy=self.scaling_factor, interpolation=cv2.INTER_AREA
        )
        self.convert_grayscale = cv2.cvtColor(
            self.frame, cv2.COLOR_RGB2GRAY
        )
        return self.convert_grayscale

    def start_frame(self):
        esc_key = int(os.getenv("ESC_KEY"))
        wait_key = int(os.getenv("WAIT_KEY"))

        while True:
            cv2.imshow(
                'Object Movement',
                self.frame_difference(
                    self.previous_frame,
                    self.current_frame,
                    self.next_frame)
                )
            self.previous_frame = self.current_frame
            self.current_frame = self.next_frame
            self.next_frame = frame.get_frame()

            key = cv2.waitKey(wait_key)
            if key == esc_key:
                break
        cv2.destroyAllWindows()


frame = Frame()
frame.start_frame()
