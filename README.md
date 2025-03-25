# Hand Gesture Mouse Control

This project utilizes OpenCV, MediaPipe, and PyAutoGUI to create a virtual mouse control system using hand gestures detected via a webcam.

## Features
- **Cursor Control:** Move the mouse pointer using the index finger.
- **Clicking:** Perform a mouse click by pinching the thumb and index finger together.
- **Scrolling:** Scroll up or down by making a fist.

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- MediaPipe
- Numpy
- PyAutoGUI

Install the dependencies using:
```bash
pip install opencv-python mediapipe numpy pyautogui
```

## How It Works
1. **Hand Detection:** Uses MediaPipe's hand detection model to find and track hand landmarks.
2. **Cursor Movement:** Maps the index finger's position to the screen dimensions using PyAutoGUI.
3. **Clicking:** Detects a pinch gesture between the thumb and index finger.
4. **Scrolling:** Detects a fist gesture to control vertical scrolling.

## Usage
- **Run the Script:**
```bash
python hand_gesture_mouse.py
```
- **Controls:**
  - Move the cursor using the index finger.
  - Pinch thumb and index finger for clicking.
  - Make a fist to activate scrolling.
  - Press 'Q' to exit.

## Customization
- Adjust the smoothing factor for smoother cursor movement.
- Modify the scroll sensitivity to match your preference.
- Change the click cooldown to avoid accidental double-clicks.

## Limitations
- Works best under good lighting conditions.
- Only supports single-hand gestures.
- May require calibration for varying screen sizes.

## Future Improvements
- Add multi-hand gesture recognition.
- Improve gesture stability for better accuracy.
- Support for right-click and drag gestures.
