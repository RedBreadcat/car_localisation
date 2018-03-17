#!/usr/bin/env python
import rospy
import signal #for sigterm
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import os
import copy
#import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
#import zipfile

#from collections import defaultdict
#from io import StringIO
from matplotlib import pyplot as plt
#from PIL import Image

class ObjectDetector:
  def __init__(self):
    signal.signal(signal.SIGINT, self.SignalHandler)
    IMAGE_SIZE = (20, 6) #inches
    plt.ion()
    self.fig = plt.figure(figsize=IMAGE_SIZE)
    self.bridge = CvBridge()
    OBJECT_DETECTION_PATH = "/home/ubuntu/git/models/research/object_detection"
    sys.path.append(OBJECT_DETECTION_PATH) #Need to append path with the object detection stuff
    print("Importing Tensorflow's Object Detector")
    from utils import label_map_util
    from utils import visualization_utils as vis_util #imports are here, since the path needs to be appended first
    self.vis_util = vis_util
    MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
    MODEL_FILE = MODEL_NAME + '.tar.gz'
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = OBJECT_DETECTION_PATH + "/data/mscoco_label_map.pbtxt"
    NUM_CLASSES = 90

    print("Opening model tar")
    tar_file = tarfile.open("/home/ubuntu/slam/object_detection/" + MODEL_FILE)
    for file in tar_file.getmembers():
      print("Loaded: " + file.name)
      file_name = os.path.basename(file.name)
      if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())

    print("Forming input graph")
    input_graph = tf.Graph()
    with tf.Session(graph=input_graph):
        score = tf.placeholder(tf.float32, shape=(None, 1917, 90), name="Postprocessor/convert_scores")
        expand = tf.placeholder(tf.float32, shape=(None, 1917, 1, 4), name="Postprocessor/ExpandDims_1")
        for node in input_graph.as_graph_def().node:
            if node.name == "Postprocessor/convert_scores":
                score_def = node
            if node.name == "Postprocessor/ExpandDims_1":
                expand_def = node

    print("Forming detection graph")
    self.detection_graph = tf.Graph()
    with self.detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        dest_nodes = ['Postprocessor/convert_scores','Postprocessor/ExpandDims_1']

        edges = {}
        name_to_node_map = {}
        node_seq = {}
        seq = 0
        for node in od_graph_def.node:
          n = self._node_name(node.name)
          name_to_node_map[n] = node
          edges[n] = [self._node_name(x) for x in node.input]
          node_seq[n] = seq
          seq += 1

        for d in dest_nodes:
          assert d in name_to_node_map, "%s is not in graph" % d

        nodes_to_keep = set()
        next_to_visit = dest_nodes[:]
        while next_to_visit:
          n = next_to_visit[0]
          del next_to_visit[0]
          if n in nodes_to_keep:
            continue
          nodes_to_keep.add(n)
          next_to_visit += edges[n]

        nodes_to_keep_list = sorted(list(nodes_to_keep), key=lambda n: node_seq[n])

        nodes_to_remove = set()
        for n in node_seq:
          if n in nodes_to_keep_list: continue
          nodes_to_remove.add(n)
        nodes_to_remove_list = sorted(list(nodes_to_remove), key=lambda n: node_seq[n])

        keep = graph_pb2.GraphDef()
        for n in nodes_to_keep_list:
          keep.node.extend([copy.deepcopy(name_to_node_map[n])])

        remove = graph_pb2.GraphDef()
        remove.node.extend([score_def])
        remove.node.extend([expand_def])
        for n in nodes_to_remove_list:
          remove.node.extend([copy.deepcopy(name_to_node_map[n])])

        with tf.device('/gpu:0'):
          tf.import_graph_def(keep, name='')
        with tf.device('/cpu:0'):
          tf.import_graph_def(remove, name='')

    print("Loading labels from: " + PATH_TO_LABELS)
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    self.category_index = label_map_util.create_category_index(categories)
    self.image_sub = rospy.Subscriber("/camera/left/image_raw", Image, self.ImageCallback)  #Subscribe needs to go at end, because rospy spins at all times
    self.imgStack = []
    self.sess = tf.Session(graph=self.detection_graph,config=tf.ConfigProto(allow_soft_placement=True))
    self.doLoop = True
    print("Object detector started")
   
  def ImageCallback(self, data):
    self.imgStack.append(data)

  def PopLoop(self):
    while self.doLoop:
      try:
        data = self.imgStack.pop()
        try:
          convertS = time.clock()
          image_np_big = self.bridge.imgmsg_to_cv2(data, "bgr8")  #in cv2, numpy arrays and Mat are super compatible
          image_np = cv2.resize(image_np_big, (0,0), fx=0.5, fy=0.5)
          image_np_expanded = np.expand_dims(image_np, axis=0)
          print('convert: ' + str(time.clock() - convertS))
        except CvBridgeError as e:
          print(e)
        with self.detection_graph.as_default():
          setupS = time.clock()
          image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
          score_out = self.detection_graph.get_tensor_by_name('Postprocessor/convert_scores:0')
          expand_out = self.detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1:0')
          score_in = self.detection_graph.get_tensor_by_name('Postprocessor/convert_scores_1:0')
          expand_in = self.detection_graph.get_tensor_by_name('Postprocessor/ExpandDims_1_1:0')
          detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
          detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
          detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
          num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
          # Actual detection.
          print('setup: ' + str(time.clock() - setupS))
          runS = time.clock()
          print('A')
          (score, expand) = self.sess.run([score_out, expand_out], feed_dict={image_tensor: image_np_expanded})
          print('B')
          (boxes, scores, classes, num) = self.sess.run([detection_boxes, detection_scores, detection_classes, num_detections], feed_dict={score_in:score, expand_in: expand})
          print('run: ' + str(time.clock() - runS))
          # Visualization of the results of a detection.
          visS = time.clock()
          self.vis_util.visualize_boxes_and_labels_on_image_array(image_np, np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores), self.category_index, use_normalized_coordinates=True, line_thickness=8)
          print('vis: ' + str(time.clock() - visS))
          print("Plotting " + str(data.header.seq))
          imshowS = time.clock()
          plt.imshow(image_np)
          print('setup: ' + str(time.clock() - imshowS))
          layoutS = time.clock()
          self.fig.tight_layout()
          print('layout: ' + str(time.clock() - layoutS))
          drawS = time.clock()
          plt.draw()
          print('draw: ' + str(time.clock() - drawS))
          plt.pause(0.001)
      except:
        pass

  def SignalHandler(self, signal, frame):
    self.doLoop = False

  def Shutdown(self):
    print("Shutting down object detector")
    self.sess.close()

  def _node_name(self, n):
    if n.startswith("^"):
      return n[1:]
    else:
      return n.split(":")[0]


def main():
  rospy.init_node('objectdetector')
  od = ObjectDetector()
  od.PopLoop()
  od.Shutdown()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()
