# Real-time Multi-Facial-Pose-Tracking
>Detecting Faces in the ROI,
>
>Focusing on the face close to the center of the video frame, 
>
>Detecting 68 Facial landmarks, 
>
>Detecting and Tracking facial pose
>

# Setup

To test the code, first make a directory and download the model files in it:

[haarcascade_frontalface_default.xml](https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml) AND [lbfmodel.yaml](https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml)

Make sure to keep the names consitent. Input the directory path to run the code as follows:

## Usage
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

