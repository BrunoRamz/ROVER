import cv2
import os


from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())

ESC_KEY = int(os.getenv("ESC_KEY"))
WAIT_KEY = int(os.getenv("WAIT_KEY"))
SCALING_FACTOR = float(os.getenv("SCALING_FACTOR"))
DEVICE_INDEX = int(os.getenv("DEVICE_INDEX"))
PATH_HAAR_CASCADE = os.getenv("PATH_HAAR_CASCADE")
HAARCASCADE_FRONTAL_FACE = os.getenv("HAARCASCADE_FRONTAL_FACE")
HAARCASCADE_EYES = os.getenv("HAARCASCADE_EYES")


class FaceEyesDetection:
    def __init__(self):
        try:
            self.face_cascade_file = \
                f'{PATH_HAAR_CASCADE}{HAARCASCADE_FRONTAL_FACE}'
            self.eyes_cascade_file = \
                f'{PATH_HAAR_CASCADE}{HAARCASCADE_EYES}'
            self.face_cascade = cv2.CascadeClassifier(self.face_cascade_file)
            self.eyes_cascade = cv2.CascadeClassifier(self.eyes_cascade_file)
        except IOError:
            print(
                "Unable to load the face and eyes cascade classifier xml file."
            )
        except FileNotFoundError:
            print("File does not exist.")
        self.capture = cv2.VideoCapture(DEVICE_INDEX)

    def start_detection(self):
        while True:
            _, self.frame = self.capture.read()
            self.resize_image_frame = cv2.resize(
                self.frame, None, fx=SCALING_FACTOR,
                fy=SCALING_FACTOR, interpolation=cv2.INTER_AREA
            )
            self.frame_gray_convertion = cv2.cvtColor(
                self.frame, cv2.COLOR_BGR2GRAY
            )
            self.face_detector = self.face_cascade.detectMultiScale(
                self.frame_gray_convertion, 1.3, 5
            )
            for (
                self.x, self.y,
                self.width, self.height
            ) in self.face_detector:
                self.rectangle_width = self.x + self.width
                self.rectangle_height = self.y + self.height
                cv2.rectangle(
                    self.frame, (self.x, self.y),
                    (self.rectangle_width, self.rectangle_height), (255, 0, 0),
                    3
                )
                self.roi_gray = self.frame_gray_convertion[
                    self.y:self.rectangle_height,
                    self.x:self.rectangle_width
                ]
                self.roi_color = self.frame[
                    self.y:self.rectangle_height,
                    self.x:self.rectangle_width
                ]
                self.eyes = self.eyes_cascade.detectMultiScale(self.roi_gray)

                for (
                    self.x_eye, self.y_eye,
                    self.eye_width, self.eye_height
                ) in self.eyes:
                    self.x_eye_center = int(self.x_eye + 0.5*self.eye_width)
                    self.y_eye_center = int(self.y_eye + 0.5*self.eye_height)
                    self.center = (self.x_eye_center, self.y_eye_center)
                    self.radius = int(0.3*(self.eye_width + self.eye_height))
                    self.color = (0, 0, 255)
                    self.thickness = 3
                    cv2.circle(
                        self.roi_color, self.center, self.radius,
                        self.color, self.thickness
                    )
    
            cv2.imshow("Face & Eye Detector", self.frame & self.frame)
            key = cv2.waitKey(WAIT_KEY)
            if key == ESC_KEY:
                break

        self.capture.release()
        cv2.destroyAllWindows()


face_eyes_detection = FaceEyesDetection()
face_eyes_detection.start_detection()
