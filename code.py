from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():

	def __init__(self, maxDisappeared=50):

        ###Assign ID to objects
		self.nextObjectID = 0 ###Assign ID to objects

        ##dictionary that has ID as key and the centroid (x-y coordinates) as vlaue
		self.objects = OrderedDict()

        #### Dictionary that marks the number of consecutive frames (value) a
        ## particular object ID (key) has been marked as “lost
		self.disappeared = OrderedDict()

        #The number of consecutive frames an object is allowed to be marked as “lost”
		self.maxDisappeared = maxDisappeared


    def register(self, centroid):

        #centroid is added to the objects
        # dictionary using the next available object ID.
        self.objects[self.nextObjectID] = centroid

        #The number of times an object has disappeared
        # is initialized to  0  in the disappeared  dictionary
        self.disappeared[self.nextObjectID] = 0
        #increment the nextObjectID  so that if a new object comes into view,
        # it will be associated with a unique ID
        self.nextObjectID += 1


    def deregister(self, objectID):
       # to deregister an object ID we delete the object ID from
       # both of our respective dictionaries
       del self.objects[objectID]
       del self.disappeared[objectID]


    def update(self, rects):

        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:

            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):

        # use the bounding box coordinates to derive the centroid
        cX = int((startX + endX) / 2.0)

        cY = int((startY + endY) / 2.0)
        inputCentroids[i] = (cX, cY)

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])






