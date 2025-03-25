import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

# Initialize MediaPipe solutions
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Screen dimensions for mapping
screen_width, screen_height = pyautogui.size()
pyautogui.FAILSAFE = False  # Disables fail-safe to prevent cursor from getting stuck

# Smoothing parameters
smoothing = 5
prev_x, prev_y = 0, 0

# Gesture state
is_clicking = False
is_scrolling = False
last_click_time = 0
scroll_start_y = 0
click_cooldown = 0.5  # Seconds between clicks to prevent accidental double-clicks

# Scroll parameters
scroll_threshold = 50  # Minimum vertical movement to trigger scroll
scroll_sensitivity = 1  # Adjust scroll speed (lower = slower)

# Initialize webcam
cap = cv2.VideoCapture(0)

def is_fist(hand_landmarks):
    """
    Detect if hand is in a fist position
    Check if all fingers are bent by comparing their tip and base positions
    """
    # Get landmarks for finger tips and base joints
    finger_tips = [
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    
    finger_mcp = [
        mp_hands.HandLandmark.THUMB_MCP,
        mp_hands.HandLandmark.INDEX_FINGER_MCP,
        mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
        mp_hands.HandLandmark.RING_FINGER_MCP,
        mp_hands.HandLandmark.PINKY_MCP
    ]
    
    # Check if each finger tip is below its base joint
    fist_count = 0
    for tip, base in zip(finger_tips, finger_mcp):
        tip_landmark = hand_landmarks.landmark[tip]
        base_landmark = hand_landmarks.landmark[base]
        
        # If tip is below base, finger is bent
        if tip_landmark.y > base_landmark.y:
            fist_count += 1
    
    # If 4 or 5 fingers are bent, consider it a fist
    return fist_count >= 4

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to read from webcam")
        continue
    
    # Flip the image horizontally for a mirror effect
    image = cv2.flip(image, 1)
    image_height, image_width, _ = image.shape
    
    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image and find hands
    results = hands.process(image_rgb)
    
    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get index finger tip coordinates for cursor movement
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_x = int(index_finger_tip.x * image_width)
            index_y = int(index_finger_tip.y * image_height)
            
            # Get thumb tip coordinates
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_x = int(thumb_tip.x * image_width)
            thumb_y = int(thumb_tip.y * image_height)
            
            # Get middle finger tip for stabilization
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middle_x = int(middle_finger_tip.x * image_width)
            middle_y = int(middle_finger_tip.y * image_height)
            
            # Calculate distance between thumb and index finger (for pinch detection)
            pinch_distance = np.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
            
            # Map index finger position to screen coordinates
            # Using the middle point between index and middle finger for more stability
            cursor_x = (index_x + middle_x) // 2
            cursor_y = (index_y + middle_y) // 2
            
            mouse_x = np.interp(cursor_x, [50, image_width-50], [0, screen_width])
            mouse_y = np.interp(cursor_y, [50, image_height-50], [0, screen_height])
            
            # Apply smoothing
            smoothed_x = prev_x + (mouse_x - prev_x) / smoothing
            smoothed_y = prev_y + (mouse_y - prev_y) / smoothing
            
            # Move mouse based on index finger position
            pyautogui.moveTo(smoothed_x, smoothed_y)
            prev_x, prev_y = smoothed_x, smoothed_y
            
            # Draw cursor indicator
            cv2.circle(image, (cursor_x, cursor_y), 10, (0, 255, 0), -1)
            
            # Scroll Detection with Fist
            if is_fist(hand_landmarks):
                if not is_scrolling:
                    # Start scrolling
                    is_scrolling = True
                    scroll_start_y = middle_y
                else:
                    # Calculate scroll amount
                    scroll_diff = middle_y - scroll_start_y
                    
                    # Scroll only if movement exceeds threshold
                    if abs(scroll_diff) > scroll_threshold:
                        # Negative scroll_diff means scrolling down
                        scroll_amount = int(scroll_diff * scroll_sensitivity)
                        pyautogui.scroll(-scroll_amount)
                        
                        # Reset scroll start point
                        scroll_start_y = middle_y
                
                # Visualize scroll gesture
                cv2.putText(image, "SCROLLING", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            else:
                is_scrolling = False
            
            # Detect pinch gesture for clicking
            # The threshold value may need adjustment based on your hand size
            pinch_threshold = 30  # Lower value = tighter pinch required
            
            current_time = time.time()
            if pinch_distance < pinch_threshold and not is_clicking and (current_time - last_click_time) > click_cooldown:
                # Perform the click
                pyautogui.click()
                is_clicking = True
                last_click_time = current_time
                cv2.putText(image, "CLICK!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            elif pinch_distance >= pinch_threshold:
                is_clicking = False
    
    # Display control instructions
    cv2.putText(image, "Move: Index finger", (image_width-300, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(image, "Click: Pinch thumb & index", (image_width-300, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(image, "Scroll: Make a Fist", (image_width-300, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(image, "Press Q to quit", (image_width-300, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Display the image
    cv2.imshow('Hand Gesture Mouse Control', image)
    
    # Press 'q' to exit
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
