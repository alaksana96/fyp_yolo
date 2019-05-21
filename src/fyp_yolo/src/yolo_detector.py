#!/usr/bin/env python

import sys, os

''' 
    Adding Darknet to Python Path
    This is done so we can access the darknet.py functions
'''
absFilePath = os.path.abspath(__file__)
fileDir     = os.path.dirname(absFilePath)
parentDir   = os.path.dirname(fileDir)

darknetPath = os.path.join(parentDir, 'src/darknet')
darknetPythonPath = os.path.join(darknetPath, 'python')

print('Adding Darknet to Python Path by inserting: {}'.format(darknetPythonPath))
sys.path.insert(0, darknetPythonPath)

import darknet as dn

import rospy
from   sensor_msgs.msg import CompressedImage
import cv_bridge as bridge

from fyp_yolo.msg import BoundingBox, CompressedImageAndBoundingBoxes

import cv2
import numpy as np

import pdb


class yolo_detector:

    def __init__(self, debug = 0, flip = False):
        
        self.debug = debug

        self.flip   = flip

        self.bridge = bridge.CvBridge()

        dn.set_gpu(0)

        self.net  = dn.load_net(os.path.join(darknetPath, 'cfg/yolov3-tiny-crowdhuman.cfg'),
                        os.path.join(darknetPath, 'weights/yolov3-tiny-crowdhuman_30000.weights'),
                        0)
        self.meta = dn.load_meta(os.path.join(darknetPath, 'cfg/yolo_crowdhuman.data'))


        self.subscriber = rospy.Subscriber('image_transport/compressed', 
                                            CompressedImage, 
                                            self.callback, 
                                            queue_size = 10 )                                                

        self.publisherImageAndDetections = rospy.Publisher('yolo_detector/output/compresseddetections',
                                                           CompressedImageAndBoundingBoxes,
                                                           queue_size = 10)


    def callback(self, ros_data):
        img = cv2.imdecode(np.fromstring(ros_data.data, np.uint8), 1)
        
        if self.flip:
            img = cv2.flip(img, 1)

        detections = dn.detect(self.net, self.meta, img)
        
        msgDetections = []

        for detected in detections:

            x = int(detected[2][0])
            y = int(detected[2][1])

            w = int(detected[2][2] / 2)
            h = int(detected[2][3] / 2 )
            
            msgDetection = BoundingBox()
            
            msgDetection.Class       = detected[0]
            msgDetection.probability = detected[1]
            msgDetection.xmin        = x - w
            msgDetection.ymin        = y - h
            msgDetection.xmax        = x + w
            msgDetection.ymax        = y + h

            msgDetections.append(msgDetection)

        # Publish detections
        msgImageAndDetections = CompressedImageAndBoundingBoxes()

        if self.flip:
            compressed                 = self.bridge.cv2_to_compressed_imgmsg(img, 'jpg')
            msgImageAndDetections.data = compressed.data
        else:
            msgImageAndDetections.data = ros_data.data

        msgImageAndDetections.format         = 'jpg'
        msgImageAndDetections.bounding_boxes = msgDetections

        self.publisherImageAndDetections.publish(msgImageAndDetections)


        # Display Bounding Boxes & Print Detections
        if self.debug > 0:
            print(detections)
            self.displayDetections(detections, img)


    def displayDetections(self, detections, img):

        for detected in detections:
            # x,y co-ordinates are the centre of the object
            x = int(detected[2][0])
            y = int(detected[2][1])

            w = int(detected[2][2] / 2)
            h = int(detected[2][3] / 2 )

            # Draw bounding box and label
            cv2.rectangle(img, (x - w , y - h), (x + w, y + h), (0,255,0))
            labelText = '{}: {:.4f}'.format(detected[0], detected[1])
            cv2.putText(img, labelText, (x, y - h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0))

        cv2.imshow('yolo: yolo_detector.py', img)
        cv2.waitKey(3)


def main(args):
    
    yd = yolo_detector(debug = 1, flip = False)
    rospy.init_node('yolo_detector', anonymous=True)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print('Shutting down Yolo Detector')
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)

