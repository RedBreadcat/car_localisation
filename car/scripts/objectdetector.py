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
from matplotlib import pyplot as plt
#from PIL import Image

class ObjectDetector:
  def __init__(self):
    IMAGE_SIZE = (20, 6) #inches
    plt.ion()
    self.fig = plt.figure(figsize=IMAGE_SIZE)
    self.bridge = CvBridge()
    OBJECT_DETECTION_PATH = "/home/roby/git/models/research/object_detection"
    sys.path.append(OBJECT_DETECTION_PATH) #Need to append path with the object detection stuff
    print("Importing Tensorflow's Object Detector")
    from utils import label_map_util
    from utils import visualization_utils as vis_util #imports are here,since the path needs to be appended first
    self.vis_util = vis_util
    MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
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

    print("Forming detection graph")
    self.detection_graph = tf.Graph()
    with self.detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    print("Loading labels from: " + PATH_TO_LABELS)
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    self.category_index = label_map_util.create_category_index(categories)
    self.image_sub = rospy.Subscriber("/camera/left/image_raw", Image, self.ImageCallback)  #Subscribe needs to go at end, because rospy spins at all times
    self.imgStack = []
    print("Object detector started")
   
  def ImageCallback(self, data):
    self.imgStack.append(data)

  def PopLoop(self):
    while True:
      try:
        data = self.imgStack.pop()
        try:
          image_np = self.bridge.imgmsg_to_cv2(data, "bgr8")  #in cv2, numpy arrays and Mat are super compatible
        except CvBridgeError as e:
          print(e)
        with self.detection_graph.as_default():
          with tf.Session(graph=self.detection_graph) as sess:
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            self.vis_util.visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), self.category_index, use_normalized_coordinates=True, line_thickness=8)
            print("Plotting " + str(data.header.seq))
            plt.imshow(image_np)
            self.fig.tight_layout()
            plt.draw()
            plt.pause(0.001)
      except:
        pass


#TODO: Want to make callback put images into a stack, and then main thread empties the stack to process the most recent image


def main():
  rospy.init_node('objectdetector')
  od = ObjectDetector()
  od.PopLoop()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()