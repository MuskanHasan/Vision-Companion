import RPi.GPIO as GPIO
import time
from time import sleep
from gpiozero import Buzzer
import argparse
import pyttsx3
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils import visualize
from picamera2 import Picamera2
import threading

# Initialize GPIO settings
GPIO.setmode(GPIO.BOARD)
buzzer = Buzzer(17)
trig = 18
echo = 16
GPIO.setup(trig, GPIO.OUT)
GPIO.setup(echo, GPIO.IN)

# Global variables for object detection
COUNTER, FPS = 0, 0
START_TIME = time.time()
picam2 = Picamera2()
picam2.preview_configuration.main.size = (320, 240)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

detection_result_list = []

def control_buzzer(duration):
    """Control the buzzer."""
    buzzer.on()
    print("Buzzer ON")
    sleep(duration)
    buzzer.off()
    print("Buzzer OFF")

def save_result(result: vision.ObjectDetectorResult, unused_output_image: mp.Image, timestamp_ms: int):
    """Callback function to save detection results."""
    global FPS, COUNTER, START_TIME
    try:
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()
        detection_result_list.append(result)
        COUNTER += 1
    except Exception as e:
        print(f"Error in save_result: {e}")

def run_object_detection(model: str, max_results: int, score_threshold: float, camera_id: int, width: int, height: int) -> None:
    """Run object detection on images acquired from the camera."""
    global detection_result_list, stop_threads
    row_size = 50
    left_margin = 24
    text_color = (0, 0, 0)
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10
    detection_result_list.clear()

    base_options = python.BaseOptions(model_asset_path=model)
    options = vision.ObjectDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        max_results=max_results,
        score_threshold=score_threshold,
        result_callback=save_result
    )
    detector = vision.ObjectDetector.create_from_options(options)

    while not stop_threads:
        im = picam2.capture_array()
        image = cv2.resize(im, (320, 240))
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        detector.detect_async(mp_image, time.time_ns() // 2_000_000)

        fps_text = 'FPS = {:.1f}'.format(FPS)
        text_location = (left_margin, row_size)
        current_frame = image
        cv2.putText(current_frame, fps_text, text_location, cv2.FONT_HERSHEY_DUPLEX, font_size, text_color, font_thickness, cv2.LINE_AA)

        if detection_result_list:
            if len(detection_result_list[0].detections) > 0:
                engine = pyttsx3.init()
                engine.say(detection_result_list[0].detections[0].categories[0].category_name + " is 20cm Away")
                engine.runAndWait()
            current_frame = visualize(current_frame, detection_result_list[0])
            detection_result_list.clear()
        else:
            current_frame = image

        if current_frame is not None:
            cv2.imshow('object_detection', current_frame)

        if cv2.waitKey(1) == 27:
            break

    detector.close()
    picam2.close()
    cv2.destroyAllWindows()

def read_ultrasonic():
    """Read the ultrasonic sensor and trigger object detection if within range."""
    global stop_threads
    try:
        while not stop_threads:
            GPIO.output(trig, True)
            time.sleep(0.00001)
            GPIO.output(trig, False)

            pulse_start = time.time()
            while GPIO.input(echo) == 0:
                pulse_start = time.time()

            pulse_end = time.time()
            while GPIO.input(echo) == 1:
                pulse_end = time.time()

            pulse_duration = pulse_end - pulse_start
            speed_of_sound = 34300
            distance = (pulse_duration * speed_of_sound) / 2
            dist = round(distance, 2)

            if dist > 2 and dist < 20:
                print(f"Distance: {dist} cm")
                print("**Object Detected!**")
                control_buzzer(1)
            else:
                print("**Out of Range**")

            time.sleep(3)

    except KeyboardInterrupt:
        stop_threads = True
        print("Measurement stopped by user")
        GPIO.cleanup()
        buzzer.close()

stop_threads = False
ultrasonic_thread = threading.Thread(target=read_ultrasonic)
ultrasonic_thread.start()

try:
    run_object_detection(model='efficientdet_lite0.tflite', max_results=1, score_threshold=0.25, camera_id=0, width=320, height=240)
except KeyboardInterrupt:
    stop_threads = True
    ultrasonic_thread.join()
    GPIO.cleanup()
    buzzer.close()
