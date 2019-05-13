#!/usr/bin/env python

import sys, os

''' 
    Adding Darknet to Python Path
    This is done so we can access the darknet.py functions
'''
absFilePath = os.path.abspath(__file__)
fileDir     = os.path.dirname(absFilePath)
parentDir   = os.path.dirname(fileDir)

darknetPath = os.path.join(parentDir, 'src/darknet/python')

print('Adding Darknet to Python Path by inserting: {}'.format(darknetPath))
sys.path.insert(0, darknetPath)

import darknet as dn

import rospy
import std_msgs.msg 

import cv2

# class yolo_detector:

#     # stuff happens
