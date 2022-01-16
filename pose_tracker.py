### Submission ready
### Question 1-5, Detect face, detect multifaces, detect landmarks, detect pose, track post

import os
import sys
import cv2
import argparse
import numpy as np
import mediapipe as mp
from utils import get_roi, get_valid_bboxes, get_focused_box, disp_msg
if __name__ == '__main__':

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True, type=str,
                    help="path to directory containing  haarcascade_frontalface_default.xml and lbfmodel.yaml")
    ap.add_argument("-i", "--input", required=False, type=str,
                    help="path to input video or image")
    ap.add_argument("-d", "--detectconf", required=False, type=float, default=0.5,
                    help="minimum confidence for detection")
    ap.add_argument("-t", "--trackconf", required=False, type=float, default=0.5,
                    help="minimum confidence for tracking")

    args = vars(ap.parse_args())

    # Pre-trained model files
    direc = args['model']
    haarcascade = 'haarcascade_frontalface_default.xml'
    LBFmodel = 'lbfmodel.yaml'

    ROI_CUT = 0.1  # area of ROI is ~80% the area of image
    DETECT_CONF = args['detectconf']
    TRACK_CONF = args['trackconf']
    if args['input']:
        input_vid = args['input']
    else:
        input_vid = 0

    disp_msg()  # print user greeting

    # create an instance of Cascade Classifier for face detection
    if haarcascade in os.listdir(direc):
        classifier = cv2.CascadeClassifier(os.path.join(direc, haarcascade))
    else:
        sys.exit("Face detection model file not found")

    # create an instance of the Facial landmark Detector
    if LBFmodel in os.listdir(direc):
        landmark_detector = cv2.face.createFacemarkLBF()
        landmark_detector.loadModel(os.path.join(direc, LBFmodel))
    else:
        sys.exit("LBF model file not found")

    # create an instance of Face Mesh classifier for pose detection and tracking
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=DETECT_CONF, min_tracking_confidence=TRACK_CONF)

    cap = cv2.VideoCapture(input_vid)
    while cap.isOpened():
        # load the frame
        success, frame = cap.read()
        raw_img = cv2.flip(frame, 1)  # mirror image for webcam #TODO add condition for images/videos
        rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

        # define the ROI
        roi = get_roi(raw_img, ROI_CUT)

        # perform face detection
        unfiltered_bboxes = classifier.detectMultiScale(gray_img)

        # eliminate invalid boxes
        bboxes = get_valid_bboxes(unfiltered_bboxes, roi)

        if len(bboxes) > 0:
            # find the bounding box to focus
            focused_box = get_focused_box(bboxes, np.shape(gray_img))

            # draw the bounding box for each detected face
            for box in bboxes:
                # extract
                (x, y, width, height) = box
                x2, y2 = x + width, y + height
                # draw a rectangle over the pixels
                if np.array_equal(box, focused_box[0]):
                    cv2.rectangle(raw_img, (x, y), (x2, y2), (0, 255, 0), 1)
                else:
                    cv2.rectangle(raw_img, (x, y), (x2, y2), (0, 0, 255), 1)

            # Detect landmarks on gray scale image
            _, landmarks = landmark_detector.fit(gray_img, bboxes)
            if len(landmarks) > 0:
                for landmark in landmarks:
                    for x_, y_ in landmark[0]:
                        cv2.circle(raw_img, (round(x_), round(y_)), 1, (255, 255, 255), 1)

            # To improve performance
            rgb_img.flags.writeable = False

            # Detect Face mesh landmarks
            results = face_mesh.process(rgb_img)

            # To improve performance
            rgb_img.flags.writeable = True

            img_h, img_w, img_c = raw_img.shape
            face_3d = []
            face_2d = []

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                            if idx == 1:
                                nose_2d = (lm.x * img_w, lm.y * img_h)
                                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                            x, y = int(lm.x * img_w), int(lm.y * img_h)

                            # Get the 2D Coordinates
                            face_2d.append([x, y])

                            # Get the 3D Coordinates
                            face_3d.append([x, y, lm.z])

                            # Convert it to the NumPy array
                    face_2d = np.array(face_2d, dtype=np.float64)

                    # Convert it to the NumPy array
                    face_3d = np.array(face_3d, dtype=np.float64)

                    # The camera matrix
                    focal_length = 1 * img_w

                    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                           [0, focal_length, img_w / 2],
                                           [0, 0, 1]])

                    # The Distance Matrix
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    # Solve PnP
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                    # Get rotational matrix
                    rmat, jac = cv2.Rodrigues(rot_vec)

                    # Get angles
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                    # Get the y rotation degree
                    x = angles[0] * 360
                    y = angles[1] * 360

                    # Display the nose direction
                    nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                    p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))

                    cv2.line(raw_img, p1, p2, (255, 0, 0), 2)

        # draw ROI
        cv2.rectangle(raw_img, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (255, 0, 0), 1)

        # show the image
        cv2.imshow('face detection', raw_img)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
