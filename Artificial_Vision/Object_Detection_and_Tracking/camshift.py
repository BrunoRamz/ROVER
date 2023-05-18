import cv2
import numpy as np
import os


from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())


class ObjectTracker:
    def __init__(self):
        self.capture = cv2.VideoCapture(int(os.getenv("DEVICE_INDEX")))
        _, self.frame = self.capture.read()
        self.scaling_factor = float(os.getenv("SCALING_FACTOR"))
        self.resize_image = cv2.resize(
            self.frame, None, fx=self.scaling_factor,
            fy=self.scaling_factor, interpolation=cv2.INTER_AREA
        )
        self.region_selection = None
        self.drag_start = None
        self.tracking_state = 0
        cv2.namedWindow('Object Tracker')
        cv2.setMouseCallback('Object Tracker', self.track_mouse_event)

    def track_mouse_event(self, mouse_event, x, y, flags, param):
        x, y = np.int16([x, y])

        if mouse_event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.tracking_state = 0

        if self.drag_start:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                height, width = self.frame.shape[:2]
                x_inital, y_initial = self.drag_start
                x_maximum, y_maximum = np.maximum(
                    0, np.minimum([x_inital, y_initial], [x, y])
                )
                x_minimum, y_minimum = np.minimum(
                    [width, height], np.maximum([x_inital, y_initial], [x, y])
                )
                self.region_selection = None

                x_difference = x_minimum - x_maximum
                y_difference = y_minimum - y_maximum
                if x_difference > 0 and y_difference > 0:
                    self.region_selection = (
                        x_maximum, y_maximum, x_minimum, y_minimum
                    )
            else:
                self.drag_start = None
                if self.region_selection is not None:
                    self.tracking_state = 1

    def start_tracking(self):
        esc_key = int(os.getenv("ESC_KEY"))
        wait_key = int(os.getenv("WAIT_KEY"))

        while True:
            _, self.frame = self.capture.read()
            self.resize_image = cv2.resize(
                self.frame, None, fx=self.scaling_factor,
                fy=self.scaling_factor, interpolation=cv2.INTER_AREA
            )
            frame_copy = self.frame.copy()
            hsv_convertion = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(
                hsv_convertion, np.array((0., 60., 32.)),
                np.array((180., 255., 255.))
            )
            if self.region_selection:
                (
                    x_maximum, y_maximum,
                    x_minimum, y_minimum
                ) = self.region_selection
                x_difference = x_minimum - x_maximum
                y_difference = y_minimum - y_maximum
                self.track_window = (
                    x_maximum, y_maximum,
                    x_difference, y_difference
                )
                hsv_interest_region = hsv_convertion[
                    y_maximum:y_minimum, x_maximum:x_minimum
                ]
                mask_interest_region = mask[
                    y_maximum:y_minimum, x_maximum:x_minimum
                ]
                histogram = cv2.calcHist(
                    [hsv_interest_region], [0], mask_interest_region,
                    [16], [0, 180]
                )
                cv2.normalize(histogram, histogram, 0, 255, cv2.NORM_MINMAX)
                self.histogram = histogram.reshape(-1)
                frame_copy_interest_region = frame_copy[
                    y_maximum:y_minimum, x_maximum:x_minimum
                ]
                cv2.bitwise_not(
                    frame_copy_interest_region,
                    frame_copy_interest_region
                )
                frame_copy[mask == 0] = 0

            if self.tracking_state == 1:
                self.region_selection = None
                histogram_back_projection = cv2.calcBackProject(
                    [hsv_convertion], [0], self.histogram,
                    [0, 180], 1
                )
                histogram_back_projection &= mask
                termination_criteria = (
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    10, 1
                )
                tracking_box, self.track_window = cv2.CamShift(
                    histogram_back_projection,
                    self.track_window,
                    termination_criteria
                )
                cv2.ellipse(frame_copy, tracking_box, (0, 255, 0), 2)
            cv2.imshow('Object Tracker', frame_copy)

            key = cv2.waitKey(wait_key)
            if key == esc_key:
                break
        cv2.destroyAllWindows()


object_tracker = ObjectTracker()
object_tracker.start_tracking()
