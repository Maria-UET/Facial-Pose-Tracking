# Facial-Pose-Tracking
>Detecting Faces in the ROI,
>Focusing on the face close to the center of the video frame, 
>Detecting 68 Facial landmarks, 
>Detecting and Tracking facial pose

usage: pose_tracker.py [-h] -m MODEL [-i INPUT] [-d DETECTCONF] [-t TRACKCONF]

optional arguments:
  -h, --help            show this help message and exit

  -m MODEL, --model MODEL
                        path to directory containing haarcascade_frontalface_default.xml and lbfmodel.yaml
  
  -i INPUT, --input INPUT
                        path to input video or image

  -d DETECTCONF, --detectconf DETECTCONF
                        minimum confidence for detection

  -t TRACKCONF, --trackconf TRACKCONF
                        minimum confidence for tracking

