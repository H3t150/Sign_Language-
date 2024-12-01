from cProfile import label
import cv2
import os
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Change the name of Directory manualy each time
folder = "Data/X"
if not os.path.exists(folder):
    os.makedirs(folder)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifer = Classifier('.venv\Model\keras_model.h5', '.venv\Model\labels.txt' )

offset = 20
imgSize = 300
count = 0

labels = ["A" , "B" , " C" , "D" , "E" , "F" , "G" , "H" , "I" , "J" , "K" , "L" , "M" , "N" , "O" , "P"
          , "Q" , "R" , "S" , "T" , "U" , "V" , "W" , "X" , "Y" , "Z" ,"NEXT WORD" ]

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip the camera for a mirror effect
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white background (300x300)
        bg = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the hand region
        imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]

        # Check if imgCrop is valid (non-empty)
        if imgCrop.size > 0:
            sizeRatio = h / w
            if sizeRatio > 1:
                # If height is greater than width, resize by height
                k = imgSize / h
                newWidth = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (newWidth, imgSize))
                wGap = math.ceil((imgSize - newWidth) / 2)
                bg[:, wGap: newWidth + wGap] = imgResize
                predication, index = classifer.getPrediction(img)
                print(predication,index)
            else:
                # If width is greater than height, resize by width
                k = imgSize / w
                newHeight = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, newHeight))
                hGap = math.ceil((imgSize - newHeight) / 2)
                bg[hGap: newHeight + hGap, :] = imgResize
                predication, index = classifer.getPrediction(img)
                print(predication, index)

            # Display cropped and background images
            cv2.imshow("Image Crop", imgCrop)
            cv2.imshow("Image Background", bg)
            cv2.imshow("Image", img)
            cv2.waitKey(1)
