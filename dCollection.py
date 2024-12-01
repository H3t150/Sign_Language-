import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

# Change the name of Directory manualy each time
folder = "Data/X"
if not os.path.exists(folder):
    os.makedirs(folder)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300
count = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip the camera for a mirror effect
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Create a white background (300x300)
        bg = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the hand+ region
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
            else:
                # If width is greater than height, resize by width
                k = imgSize / w
                newHeight = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, newHeight))
                hGap = math.ceil((imgSize - newHeight) / 2)
                bg[hGap: newHeight + hGap, :] = imgResize

            # Display cropped and background images
            cv2.imshow("Image Crop", imgCrop)
            cv2.imshow("Image Background", bg)

            key = cv2.waitKey(1)
            if key == ord("s"):
                count += 1
                # Save the image with a timestamp in the folder
                image_path = f'{folder}/Image_{time.time()}.jpg'
                success = cv2.imwrite(image_path, bg)
                if success:
                    print(f"Image saved successfully as {image_path}, Count: {count}")
                else:
                    print(f"Failed to save image {image_path}")

    # Display the main image
    cv2.imshow("Image", img)

    # Exit if 'q' is pressed
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
