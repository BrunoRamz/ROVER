import cv2
import os


from dotenv import load_dotenv, find_dotenv


load_dotenv(find_dotenv())


class BackgroundSubtraction:

    def get_frame(self, cap):
        scaling_factor = float(os.getenv("SCALING_FACTOR"))
        _, self.frame = cap.read()
        self.resize_image_frame = cv2.resize(
            self.frame, None, fx=scaling_factor,
            fy=scaling_factor, interpolation=cv2.INTER_AREA
        )

        return self.resize_image_frame

    def compute_maks(self, background_subtractor, frame, learning_rate):
        mask = background_subtractor.apply(frame, learningRate=learning_rate)
        grayscale_to_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        return grayscale_to_rgb


cap = cv2.VideoCapture(int(os.getenv("DEVICE_INDEX")))
history = int(os.getenv("HISTORY"))
rate = float(os.getenv("RATE"))
esc_key = int(os.getenv("ESC_KEY"))
wait_key = int(os.getenv("WAIT_KEY"))

background_subtraction = BackgroundSubtraction()
background_subtractor = cv2.createBackgroundSubtractorMOG2()

learning_rate = rate/history

while True:
    frame = background_subtraction.get_frame(cap)
    mask = background_subtraction.compute_maks(
        background_subtractor, frame, learning_rate
    )

    cv2.imshow('Intput', frame)
    cv2.imshow('Output', mask & frame)

    key = cv2.waitKey(wait_key)
    if key == esc_key:
        break

cap.release()
cv2.destroyAllWindows()
