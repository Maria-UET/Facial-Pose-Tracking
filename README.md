# Real-time Multi-Facial-Pose-Tracking
>Detecting Faces in the ROI,
>
>Focusing on the face close to the center of the video frame, 
>
>Detecting 68 Facial landmarks, 
>
>Detecting and Tracking facial pose
>

```
usage: pose_tracker.py [-h] -m MODEL [-i INPUT] [-d DETECTCONF] [-t TRACKCONF]

optional arguments:
  -h, --help            show this help message and exit
  
  -m, --model 
                        path to directory containing haarcascade_frontalface_default.xml and lbfmodel.yaml  
  -i, --input
                        path to input video or image
  -d, --detectconf
                        minimum confidence for detection
  -t, --trackconf
                        minimum confidence for tracking```

