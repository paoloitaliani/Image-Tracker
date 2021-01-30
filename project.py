from centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2

prototxt="/Users/Niolo/Documents/Github/Image-Tracker/models/deploy.prototxt.txt"

caffemodel="/Users/Niolo/Documents/Github/Image-Tracker/models/res10_300x300_ssd_iter_140000.caffemodel"

ct = CentroidTracker()
(H, W) = (None, None)

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)