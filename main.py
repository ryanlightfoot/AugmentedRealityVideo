import cv2
import numpy as np

# Load the target image
imgTarget = cv2.imread('Targetimg.png', cv2.IMREAD_GRAYSCALE)
# You should replace 'Targetimg.png' with the path to your target image file.

# Create a SIFT detector
sift = cv2.SIFT_create()
# Find the keypoints and descriptors of the target image
kp1, des1 = sift.detectAndCompute(imgTarget, None)

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 represents the default camera (you can change it if you have multiple cameras)

# Load the video
video = cv2.VideoCapture('video.mp4')  # Replace 'video.mp4' with the path to your video file.

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the keypoints and descriptors of the current frame
    kp2, des2 = sift.detectAndCompute(gray, None)

    # Create a FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Match the descriptors of the target image and the current frame
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Extract the location of the matches
    if len(good_matches) > 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Calculate the homography matrix
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is not None:
            # Get the dimensions of the target image
            h, w = imgTarget.shape[:2]

            # Transform the corners using the homography matrix
            corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners, M)

            # Draw a square around the target image
            frame = cv2.polylines(frame, [np.int32(transformed_corners)], True, (0, 255, 0), 3)

            # Check if the video is open and play it on the identified region
            if video.isOpened():
                ret, video_frame = video.read()
                if ret:
                    # Warp the video frame onto the target image
                    warped_video_frame = cv2.warpPerspective(video_frame, M, (frame.shape[1], frame.shape[0]))
                    
                    # Overlay the video frame on the target image
                    frame = cv2.addWeighted(frame, 1, warped_video_frame, 0.7, 0)

    # Display the result
    cv2.imshow('AR Video', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera, video, and close all windows
cap.release()
video.release()
cv2.destroyAllWindows()
