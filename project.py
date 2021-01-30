from centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import datetime

prototxt = "/Users/Niolo/Documents/Github/Image-Tracker/models/deploy.prototxt.txt"

caffemodel = "/Users/Niolo/Documents/Github/Image-Tracker/models/res10_300x300_ssd_iter_140000.caffemodel"

confidence = 0.5

ct = CentroidTracker(maxDisappeared=50, maxDistance=50)
(H, W) = (None, None)

print("[INFO] loading model...")

# Opening the deep learning model
net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)  # Opening the deep

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# Allow 2 seconds for the camera to warm up
time.sleep(2.0)


def main():
    fps_start_time = datetime.datetime.now()
    fps = 0
    total_frames = 0

    while True:

        # read the next frame from the video stream and resize it
        frame = vs.read()

        # resize the frames to a fixed width (while preserving aspect ratio)
        frame = imutils.resize(frame, width=400)
        total_frames = total_frames + 1

        # if the frame dimensions are None, grab them

        (H, W) = frame.shape[:2]

        # construct a blob from the frame

        blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
                                     (104.0, 177.0, 123.0))

        # Then we pass the frame through the
        # CNN object detector to obtain predictions
        net.setInput(blob)
        detections = net.forward()

        # and initialize the list of  bounding box rectangles
        rects = []

        for i in range(0, detections.shape[2]):

            # filter out weak detections by ensuring the predicted
            # probability is greater than a minimum threshold
            if detections[0, 0, i, 2] > confidence:
                # compute the (x, y)-coordinates of the bounding box for
                # the object, then update the bounding box rectangles list
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])

                # contains the box
                rects.append(box.astype("int"))

                # draw a bounding box surrounding the object so we can
                # visualize it
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 255, 0), 2)

        # update our centroid tracker using the computed set of bounding
        # box rectangles
        objects = ct.update(rects)

        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        fps_end_time = datetime.datetime.now()
        time_diff = fps_end_time - fps_start_time
        if time_diff.seconds == 0:
            fps = 0.0
        else:
            fps = (total_frames / time_diff.seconds)

        fps_text = "FPS: {:.2f}".format(fps)

        cv2.putText(frame, fps_text, (5, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

        # show the output frame
        cv2.imshow("Application", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


main()

vs.stop()
