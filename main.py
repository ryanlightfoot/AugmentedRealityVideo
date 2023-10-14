import cv2
import numpy as np

# Load target
imgTarget = cv2.imread('Targetimg.png', cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()
# Find keypoints and descriptors of target image
kp1, des1 = sift.detectAndCompute(imgTarget, None)

# Initialize camera
cap = cv2.VideoCapture(0)

# Load video
video = cv2.VideoCapture('video.mp4')

while True:
    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(gray, None)

    # Create a FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match the descriptors target image and current frame
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
            # Get dimensions of target image
            h, w = imgTarget.shape[:2]

            # Transform corners
            corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners, M)

            # Draw a square around target image
            frame = cv2.polylines(frame, [np.int32(transformed_corners)], True, (0, 255, 0), 3)

            # Check if the video is open and play on the identified region
            if video.isOpened():
                ret, video_frame = video.read()
                if ret:
                    warped_video_frame = cv2.warpPerspective(video_frame, M, (frame.shape[1], frame.shape[0]))
                    frame = cv2.addWeighted(frame, 1, warped_video_frame, 1.0, 0)

    # Display result
    cv2.imshow('AR Video', frame)

    #Press q to end
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video.release()
cv2.destroyAllWindows()
