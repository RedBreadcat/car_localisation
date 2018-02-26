#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import os
#import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
#import zipfile

#from collections import defaultdict
#from io import StringIO
#from matplotlib import pyplot as plt
#from PIL import Image

class ObjectDetector:
  def __init__(self):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera/left/image_raw", Image, self.ImageCallback)
    OBJECT_DETECTION_PATH = "/home/roby/git/models/research/object_detection"
    sys.path.append(OBJECT_DETECTION_PATH) #Need to append path with the object detection stuff
    print("Importing Tensorflow's Object Detector")
    from utils import label_map_util
    from utils import visualization_utils as vis_util #imports are here,since the path needs to be appended first
    MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = OBJECT_DETECTION_PATH + "/data/mscoco_label_map.pbtxt"
    NUM_CLASSES = 90

    print("Opening model tar")
    tar_file = tarfile.open("/home/roby/slam/object_detection/" + MODEL_FILE)
    for file in tar_file.getmembers():
      print("Loaded: " + file.name)
      file_name = os.path.basename(file.name)
      if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())

    print("Object detector started")
   
  def ImageCallback(self, data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
   
    (rows, cols, channels) = cv_image.shape

    cv2.imshow("Image window", cv_image)
    cv2.waitKey(3)

def main():
  rospy.init_node('objectdetector')
  od = ObjectDetector()
  rospy.spin()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()