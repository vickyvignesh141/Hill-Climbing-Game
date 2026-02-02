import cv2
import mediapipe as mp

# Correct updated import
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize hand model
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# Start camera
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        print("Camera not detected")
        break

    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    results = hands.process(rgb)

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show frame
    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('v'):
        break

cap.release()
cv2.destroyAllWindows()
