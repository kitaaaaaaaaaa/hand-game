# Setup
**Python**
- OpenCV
- MediaPipe Hand Landmarker
- NumPy
- Pygame
- maybe matplotlib

`pip install opencv-python mediapipe numpy pygame`

## CV
- Webcam -> OpenCV
- MediaPipe -> hand landmarks
- Custom gesture recognition:
    - fist -> punch       (0)
    - open hand -> block  (1)
    - pinch -> grab       (2)
    - others??
- Ensure gesture cooldowns

## Game
- Basic Pygame fighting game (implement game loop)
- Connect gestures to actions
- Add health, damage, win/lose -> game incentives
- Add simple animations/sound (polish)

## Innovations
- Gesture-based movement
- Hand velocity = attack strength
- two player/two hands