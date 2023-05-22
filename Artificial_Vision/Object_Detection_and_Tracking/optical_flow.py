import cv2
import numpy as np
import os


from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())

ESC_KEY = int(os.getenv("ESC_KEY"))
WAIT_KEY = int(os.getenv("WAIT_KEY"))
SCALING_FACTOR = float(os.getenv("SCALING_FACTOR"))
DEVICE_INDEX = int(os.getenv("DEVICE_INDEX"))
TRACK_FRAMES_NUMBER = int(os.getenv("TRACK_FRAMES_NUMBER"))
FRAME_INDEX = int(os.getenv("FRAME_INDEX"))
SKIP_FACTOR = int(os.getenv("SKIP_FACTOR"))


class OpticalFLow:
    def __init__(self):
        self.capture = cv2.VideoCapture(DEVICE_INDEX)
        self.tracking_paths = []
        self.tracking_parameters = {
            "winSize": (11, 11),
            "maxLevel": 2,
            "criteria": (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10, 0.03
            )
        }
        self.frame_index = FRAME_INDEX

    def start_tracking(self):
        while True:
            _, self.frame = self.capture.read()
            self.resize_image_frame = cv2.resize(
                self.frame, None, fx=SCALING_FACTOR,
                fy=SCALING_FACTOR, interpolation=cv2.INTER_AREA
            )
            self.frame_gray_convertion = cv2.cvtColor(
                self.frame, cv2.COLOR_BGR2GRAY
            )
            self.frame_copy = self.frame.copy()

            if len(self.tracking_paths) > 0:
                self.previous_image, self.current_image = (
                    self.previous_gray_image,
                    self.frame_gray_convertion
                )
                self.feature_points = np.float32(
                    [
                        self.tp[-1] for self.tp in self.tracking_paths
                    ]
                ).reshape(-1, 1, 2)
                self.optical_flow_compute, _, _ = cv2.calcOpticalFlowPyrLK(
                    self.previous_image, self.current_image,
                    self.feature_points, None,
                    **self.tracking_parameters
                )
                self.reverse_optical_flow_compute, _, _ = \
                    cv2.calcOpticalFlowPyrLK(
                        self.current_image, self.previous_image,
                        self.optical_flow_compute, None,
                        **self.tracking_parameters
                    )
                self.feature_points_difference = abs(
                    self.feature_points - self.reverse_optical_flow_compute
                ).reshape(-1, 2).max(-1)
                self.good_points_extraction = self.feature_points_difference < 1
                self.new_tracking_paths = []

                for self.tp, (self.x, self.y), self.good_points_flag in zip(
                    self.tracking_paths,
                    self.optical_flow_compute.reshape(-1, 2),
                    self.good_points_extraction
                ):
                    if not self.good_points_flag:
                        continue
                    self.tp.append((self.x, self.y))
                    if len(self.tp) > TRACK_FRAMES_NUMBER:
                        del self.tp[0]
                    self.new_tracking_paths.append(self.tp)
                    cv2.circle(
                        self.frame_copy, (int(self.x), int(self.y)),
                        3, (0, 255, 0),
                        -1
                    )
                self.tracking_paths = self.new_tracking_paths
                cv2.polylines(
                    self.frame_copy, [
                        np.int32(self.tp) for self.tp in self.tracking_paths
                    ],
                    False,
                    (255, 0, 0)
                )

            if not self.frame_index % SKIP_FACTOR:
                self.mask = np.zeros_like(self.frame_gray_convertion)
                self.mask[:] = 255
                self.tracking_paths_array = [
                    np.int32(self.tp[-1]) for self.tp in self.tracking_paths
                ]
                for self.x, self.y in self.tracking_paths_array:
                    cv2.circle(
                        self.mask, (self.x, self.y), 6, 0, -1
                    )
                self.feature_points = cv2.goodFeaturesToTrack(
                    self.frame_gray_convertion, mask=self.mask,
                    maxCorners=500, qualityLevel=0.3,
                    minDistance=7, blockSize=7
                )

                if self.feature_points is not None:
                    self.feature_points_array = np.float32(
                        self.feature_points
                    ).reshape(-1, 2)
                    for self.x, self.y in self.feature_points_array:
                        self.tracking_paths.append([(self.x, self.y)])

            self.frame_index += 1
            self.previous_gray_image = self.frame_gray_convertion

            cv2.imshow('Optical Flow', self.frame_copy)
            key = cv2.waitKey(WAIT_KEY)
            if key == ESC_KEY:
                break

        cv2.destroyAllWindows()


optical_flow = OpticalFLow()
optical_flow.start_tracking()
