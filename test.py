
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import time
import util_visualization as utils
from utils import dataset_util
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
sys.path.append("../slim")

from utils import label_map_util

from utils import visualization_utils as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'frozen_models/frozen_V_IR_M_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'sensiac_label_map.pbtxt')

NUM_CLASSES = 2


PLOT_DIR = './out/plots'
def plot_conv_weights(weights, name, channels_all=True):
    """
    Plots convolutional filters
    :param weights: numpy array of rank 4
    :param name: string, name of convolutional layer
    :param channels_all: boolean, optional
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    plot_dir = os.path.join(PLOT_DIR, 'conv_weights')
    plot_dir = os.path.join(plot_dir, name)

    # create directory if does not exist, otherwise empty it
    utils.prepare_dir(plot_dir, empty=True)

    w_min = np.min(weights)
    w_max = np.max(weights)

    channels = [0]
    # make a list of channels if all are plotted
    if channels_all:
        channels = range(weights.shape[2])

    # get number of convolutional filters
    num_filters = weights.shape[3]

    # get number of grid rows and columns
    grid_r, grid_c = utils.get_grid_dim(num_filters)

    # create figure and axes
    fig, axes = plt.subplots(min([grid_r, grid_c]),
                             max([grid_r, grid_c]))
    # iterate channels
    for channel in channels:
        # iterate filters inside every channel
        for l, ax in enumerate(axes.flat):
            # get a single filter
            img = weights[:, :, channel, l]
            # put it on the grid
            ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='nearest', cmap='seismic')
            # remove any labels from the axes
            ax.set_xticks([])
            ax.set_yticks([])
        # save figure
        plt.savefig(os.path.join(plot_dir, '{}-{}.png'.format(name, channel)), bbox_inches='tight')

def plot_conv_output(conv_img, name):
    """
    Makes plots of results of performing convolution
    :param conv_img: numpy array of rank 4
    :param name: string, name of convolutional layer
    :return: nothing, plots are saved on the disk
    """
    # make path to output folder
    plot_dir = os.path.join(PLOT_DIR, 'conv_output')
    plot_dir = os.path.join(plot_dir, name)

    # create directory if does not exist, otherwise empty it
    utils.prepare_dir(plot_dir, empty=True)

    h = conv_img.shape[1]
    w = conv_img.shape[2]
    ave_img = np.average(conv_img,axis=3)
    ave_img = np.reshape(ave_img,(h,w))
    # w_min = np.min(conv_img)
    # w_max = np.max(conv_img)
    #
    # # get number of convolutional filters
    # num_filters = conv_img.shape[3]
    #
    # # get number of grid rows and columns
    # grid_r, grid_c = utils.get_grid_dim(num_filters)
    #
    # # create figure and axes
    # fig, axes = plt.subplots(min([grid_r, grid_c]),
    #                          max([grid_r, grid_c]))
    #
    # # iterate filters
    # for l, ax in enumerate(axes.flat):
    #     # get a single image
    #     img = conv_img[0, :, :,  l]
    #     # put it on the grid
    #     ax.imshow(img, vmin=w_min, vmax=w_max, interpolation='bicubic', cmap='Greys')
    #     # remove any labels from the axes
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    # save figure
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.axis('off')

    plt.imshow(ave_img)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    plt.savefig(os.path.join(plot_dir, '{}.png'.format(name)), bbox_inches=extent)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  PNG = np.array(image.getdata()).reshape(
      (im_height, im_width, 4)).astype(np.uint8)
  JPG = PNG[...,:3]
  return JPG

type = "3C_tank"
# name_list_path = "dataset/Train_Test/"+type+"/test.txt"
# names = dataset_util.read_examples_list(name_list_path)

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = "images/"
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}.png'.format(type))]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    for image_path in TEST_IMAGE_PATHS:
      start = time.clock()
      image = Image.open(image_path)
      print image.size
      # the array based representation of the image will be used later in order to prepare the
      # result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      for n in tf.get_default_graph().as_graph_def().node:
          print n.name
      feature_2x = detection_graph.get_tensor_by_name('FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/conv1/Relu:0')
      feature_4x = detection_graph.get_tensor_by_name('FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/block1/unit_2/bottleneck_v1/Relu:0')
      feature_6x = detection_graph.get_tensor_by_name('FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/block2/unit_3/bottleneck_v1/Relu:0')
      feature_8x = detection_graph.get_tensor_by_name('FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/block3/unit_23/bottleneck_v1/Relu:0')
      # Actual detection.
      (feature_2x,feature_4x,feature_6x,feature_8x,boxes, scores, classes, num_detections) = sess.run(
          [feature_2x,feature_4x,feature_6x,feature_8x, boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      end = time.clock()
      print end-start
      #Visualization of the layers
      plot_conv_output(feature_2x,type+"2X")
      print feature_2x.shape

      plot_conv_output(feature_4x,type+"4X")
      print feature_4x.shape

      plot_conv_output(feature_6x,type+"6X")
      print feature_6x.shape

      plot_conv_output(feature_8x,type+"8X")
      print feature_8x.shape


      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=2)
      plt.figure(figsize=IMAGE_SIZE)
      plot_dir = os.path.join(PLOT_DIR, 'detections')
      utils.prepare_dir(plot_dir, empty=True)
      fig = plt.figure()
      ax = fig.add_subplot(1, 1, 1)
      plt.axis('off')

      plt.imshow(image_np)
      extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
      plt.savefig(os.path.join(plot_dir, '{}.png'.format(type)), bbox_inches=extent)
      plt.show()