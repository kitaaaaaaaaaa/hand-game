# Import libraries
import cv2, time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import pyautogui

gesture_state = None

gesture_model = joblib.load("./dependencies/direction_model.pkl")

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        # cv2.putText(annotated_image, f"{handedness[0].category_name}",
        #             (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
        #             FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        # Draw gesture prediction on the image
        data = []
        for landmark in hand_landmarks:
            data.append(landmark.x)
            data.append(landmark.y)

        prediction = gesture_model.predict([data])[0]
        prob = gesture_model.predict_proba([data])[0]

        global gesture_state
        gesture_state = prediction
        
        category_name = {
            0: "up",
            1: "right",
            2: "down",
            3: "left",
            4: "open hand",
            5: "closed hand"
        }

        cv2.putText(annotated_image, f"{category_name[prediction]} {round(prob[prediction], 2)}",
            (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

# Initialise camera capture
cam = cv2.VideoCapture(0)
start_time = time.time()

# Setup the task
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

annotated_frame = None

# Create a rate limited function to perform a mouse left click
last_click_time = 0

def left_click():
    global last_click_time

    current_time = time.time()

    if current_time - last_click_time > 1:
        last_click_time = time.time()
        pyautogui.click()

# Create a hand landmarker instance with the live stream mode:
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global annotated_frame
    annotated_frame = draw_landmarks_on_image(output_image.numpy_view(), result)
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='./dependencies/hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    min_tracking_confidence=0.5,
    result_callback=print_result)

with HandLandmarker.create_from_options(options) as landmarker:
    while cam.isOpened():
        ok, frame = cam.read() # Read the camera data

        if not ok or frame is None:
            continue

        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_timestamp_ms = int((time.time() - start_time) * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame) # Convert the camera feed into an mp.Image type
        landmarker.detect_async(mp_image, frame_timestamp_ms) # Detect hand landmarks
        
        if annotated_frame is not None:
            cv2.imshow("Gesture detection", annotated_frame) # Show the camera in window

        # Allow quit on keypress "q"
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

        match gesture_state:
            case 0:
                pyautogui.move(0, -30)
            case 1:
                pyautogui.move(30, 0)
            case 2:
                pyautogui.move(0, 30)
            case 3:
                pyautogui.move(-30, 0)
            case 4:
                pass
            case 5:
                left_click()

    # Destroy window when complete
    cam.release()
    cv2.destroyAllWindows()