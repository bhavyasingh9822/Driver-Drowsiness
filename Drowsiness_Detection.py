from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
#import imutils
import dlib
import cv2
import numpy as np

# Initialize sound alert
mixer.init()
mixer.music.load("music.wav")

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Improved Red-Eye Detection Function
def detect_red_eye(eye_roi):
    if eye_roi.size == 0:
        return False  # Avoid errors if eye_roi is empty

    hsv = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2HSV)

    # Define a stronger red color range to minimize false positives
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create red masks
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2

    # Remove small noise using morphological operations
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    # Exclude dark areas (pupil region) to avoid false positives
    gray_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    _, pupil_mask = cv2.threshold(gray_eye, 50, 255, cv2.THRESH_BINARY_INV)  # Detect dark regions
    refined_mask = cv2.bitwise_and(red_mask, red_mask, mask=pupil_mask)  # Remove pupil areas

    # Calculate red pixel ratio
    red_pixels = cv2.countNonZero(refined_mask)
    total_pixels = eye_roi.shape[0] * eye_roi.shape[1] if eye_roi.shape[0] > 0 and eye_roi.shape[1] > 0 else 1
    red_ratio = (red_pixels / total_pixels) * 100

    return red_ratio > 10  # Adjust threshold to reduce false positives

# Initialize parameters
thresh = 0.25  # EAR threshold for drowsiness
frame_check = 20  # Frames before triggering alert
flag = 0  # Counter for drowsiness frames

# Load dlib face detector and landmark predictor
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Landmark indices for left and right eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Start webcam feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)

        # Extract eye landmarks
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Compute EAR for both eyes
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Get eye bounding boxes
        xL, yL, wL, hL = cv2.boundingRect(leftEye)
        xR, yR, wR, hR = cv2.boundingRect(rightEye)
        leftEyeROI = frame[yL:yL+hL, xL:xL+wL]
        rightEyeROI = frame[yR:yR+hR, xR:xR+wR]

        # Check for red-eye in both eyes
        left_red = detect_red_eye(leftEyeROI)
        right_red = detect_red_eye(rightEyeROI)

        # Draw eye contours
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # Check drowsiness (EAR < threshold for 20 frames)
        if ear < thresh:
            flag += 1
            if flag >= frame_check:
                cv2.putText(frame, "ALERT! DROWSINESS DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                mixer.music.play()
        else:
            flag = 0

        # Check for red-eye
        if left_red or right_red:
            cv2.putText(frame, "RED EYE DETECTED!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            mixer.music.play()

    # Show video feed
    cv2.imshow("Drowsiness & Red-Eye Detection", frame)

    # Quit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cv2.destroyAllWindows()
cap.release()
