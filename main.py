import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2, time, joblib

annotated_frame = None

gesture_state = None

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
        hand = {
            0: "left",
            1: "right"
        }

        cv2.putText(annotated_image, f"{hand[handedness[0].index]}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

        # Draw gesture prediction on the image        
        # category_name = {
        #     0: "up",
        #     1: "right",
        #     2: "down",
        #     3: "left",
        #     4: "open hand",
        #     5: "closed hand"
        # }

        # cv2.putText(annotated_image, f"{category_name[prediction]} {round(prob[prediction], 2)}",
        #     (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
        #     FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def on_result(result, output_image: mp.Image, timestamp_ms: int):
    global annotated_frame
    annotated_frame = draw_landmarks_on_image(output_image.numpy_view(), result)
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    wrist_x, wrist_y = result.hand_landmarks[0][0].x, result.hand_landmarks[0][0].y
    index_finger_mcp_x, index_finger_mcp_y = result.hand_landmarks[0][5].x, result.hand_landmarks[0][5].y
    index_finger_pip_x, index_finger_pip_y = result.hand_landmarks[0][6].x, result.hand_landmarks[0][6].y

    v50 = np.array([wrist_x - index_finger_mcp_x , wrist_y - index_finger_mcp_y])*100
    v56 = np.array([index_finger_pip_x - index_finger_mcp_x, index_finger_pip_y - index_finger_mcp_y])*100
    
    angle = np.arccos((np.dot(v50, v56))/np.linalg.norm(v50)*np.linalg.norm(v56))
    
    print(np.degrees(np.clip(np.dot(v50, v56), -1.0, 1.0))/np.linalg.norm(v50)*np.linalg.norm(v56))

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="./dependencies/hand_landmarker.task"),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=on_result,
)

with HandLandmarker.create_from_options(options) as landmarker:
    cam = cv2.VideoCapture(0)
    start_time = time.time()

    while cam.isOpened():
        ok, frame = cam.read()

        if not ok or frame is None:
            continue

        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        timestamp_ms = int((time.time() - start_time)*1000)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(mp_image, timestamp_ms)

        if annotated_frame is not None:
            cv2.imshow("Gesture detection", annotated_frame)
        else:
            cv2.imshow("Gesture detection", frame)


        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()