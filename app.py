import cv2
from cvzone.HandTrackingModule import HandDetector
from tensorflow.keras.models import load_model
import numpy as np
import time

# Load the trained model
model = load_model('C:/Users/Venkatesh/OneDrive/Desktop/Internships/Skillcraft internship/Task 4/hands/hand_gesture_model_final.keras')

# Initialize hand detector
detector = HandDetector(maxHands=1)

# Webcam setup
cap = cv2.VideoCapture(0)
imgSize = 64  # Match input size of the model

# Class labels (adjust based on your folder names)
labels = ['Fist', 'Index', 'Ok', 'Palm', 'Thumb_up', 'Thumbs_down']

while True:
    success, img = cap.read()
    if not success:
        break

    # Detect hands
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white background
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the hand region
        imgCrop = img[y:y + h, x:x + w]

        # Resize and center the hand image
        aspect_ratio = h / w
        if aspect_ratio > 1:
            k = imgSize / h
            wCal = int(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = (imgSize - wCal) // 2
            imgWhite[:, wGap:wGap + wCal] = imgResize
        else:
            k = imgSize / w
            hCal = int(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = (imgSize - hCal) // 2
            imgWhite[hGap:hGap + hCal, :] = imgResize

        # Prepare image for prediction
        imgWhite = imgWhite / 255.0  # Normalize the image
        imgWhite = np.expand_dims(imgWhite, axis=0)  # Add batch dimension

        # Make a prediction
        predictions = model.predict(imgWhite)
        classIndex = np.argmax(predictions)
        confidence = predictions[0][classIndex]

        # Display label and confidence
        label_text = f'{labels[classIndex]}: {confidence:.2f}'
        cv2.putText(img, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

    cv2.imshow("Hand Gesture Recognition", img)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
