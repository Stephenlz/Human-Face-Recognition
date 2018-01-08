# This python file will detect human face and do screenshoot then store the images to dataset
import numpy as np
import cv2
import imutils
import dlib

cap = cv2.VideoCapture(0)  # Initialize video capture.
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
pointer = 1
while cap.isOpened():
    # detect faces in the grayscale frame
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=900)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    key = cv2.waitKey(1)
    if len(rects) == 0: None
    # loop over the face detections
    for rect in rects:
        cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (255, 0, 0), 1)
        judgeImg = gray[rect.top():rect.bottom(), rect.left():rect.right()]
        cv2.imshow('Live video', frame)
        judgeImg = cv2.resize(judgeImg, (100, 100))
        if key == ord('p'):
            cv2.imwrite('photo' + str(pointer) + '.jpg', judgeImg)
            pointer += 1;


    if key == ord('q'):
        # Stop when 'q' key is pressed
        break

cv2.destroyAllWindows()  # close windows created by cv2
