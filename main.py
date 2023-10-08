import cv2
import numpy as np

# Create a VideoCapture object for the webcam (assuming it's at index 0)
cap = cv2.VideoCapture(0)

# Load an image from a file
imgTarget = cv2.imread('TargetImg.png')

# Create a VideoCapture object for the video file 'video.MP4'
myVid = cv2.VideoCapture('video.MP4')

# Read the first frame from myVid
success, imgVideo = myVid.read()

hT, wT, cT = imgTarget.shape
imgVideo = cv2.resize(imgVideo,(wT,hT))

orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(imgTarget, None)
#imgTarget = cv2.drawKeypoints(imgTarget,kp1,None)

while True:
    success, imgWebcam = cap.read()
    kp2, des2 = orb.detectAndCompute(imgWebcam, None)
    #imgWebcam = cv2.drawKeypoints(imgWebcam,kp2,None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(len(good))
    imgFeatures = cv2.drawMatches(imgTarget, kp1, imgWebcam, kp2, good, None, flags=2)

# Display the loaded image and the first frame of the video
    cv2.imshow('imgFeatures', imgFeatures)
    cv2.imshow('imgTarget', imgTarget)
    cv2.imshow('myVid', imgVideo)
    cv2.imshow('Webcam', imgWebcam)

# Wait for a key event indefinitely
    cv2.waitKey(0)

# Release the VideoCapture objects and close the OpenCV windows
    cap.release()
    myVid.release()
    cv2.destroyAllWindows()
