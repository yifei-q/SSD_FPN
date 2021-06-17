from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf
from scipy.misc import imread, imsave, imshow, imresize
import numpy as np
import imutils

from net import ssd_net

from dataset import dataset_common
from preprocessing import ssd_preprocessing
from utility import anchor_manipulator
from utility import draw_toolbox
import cv2

import networkx as nx
import math
from scipy.optimize import linear_sum_assignment

# scaffold related configuration
tf.app.flags.DEFINE_integer(
    'num_classes', 2, 'Number of classes to use in the dataset.')
# model related configuration
tf.app.flags.DEFINE_integer(
    'train_image_size', 300,
    'The size of the input image for the model to use.')
tf.app.flags.DEFINE_string(
    'data_format', 'channels_first', # 'channels_first' or 'channels_last'
    'A flag to override the data format used in the model. channels_first '
    'provides a performance boost on GPU but is not always compatible '
    'with CPU. If left unspecified, the data format will be chosen '
    'automatically based on whether TensorFlow was built for CPU or GPU.')
tf.app.flags.DEFINE_float(
    'select_threshold', 0.2, 'Class-specific confidence score threshold for selecting a box.')#SSD=0.15 ,FPN+SSD=0.05
tf.app.flags.DEFINE_float(
    'min_size', 0.03, 'The min size of bboxes to keep.')
tf.app.flags.DEFINE_float(
    'nms_threshold', 0.45, 'Matching threshold in NMS algorithm.')
tf.app.flags.DEFINE_integer(
    'nms_topk', 20, 'Number of total object to keep after NMS.')
tf.app.flags.DEFINE_integer(
    'keep_topk', 200, 'Number of total object to keep for each image before nms.')
# checkpoint related configuration
tf.app.flags.DEFINE_string(
    'checkpoint_path', './logs/',
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'model_scope', 'ssd300',
    'Model scope name used to replace the name_scope in checkpoint.')
tf.app.flags.DEFINE_float('gpu_memory_fraction', 0.1, 'GPU memory fraction to use.')

FLAGS = tf.app.flags.FLAGS
#CUDA_VISIBLE_DEVICES

def get_checkpoint():
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    return checkpoint_path

def select_bboxes(scores_pred, bboxes_pred, num_classes, select_threshold):
    selected_bboxes = {}
    selected_scores = {}
    with tf.name_scope('select_bboxes', values=[scores_pred, bboxes_pred]):
        for class_ind in range(1, num_classes):
            class_scores = scores_pred[:, class_ind]

            select_mask = class_scores > select_threshold
            select_mask = tf.cast(select_mask, tf.float32)
            selected_bboxes[class_ind] = tf.multiply(bboxes_pred, tf.expand_dims(select_mask, axis=-1))
            selected_scores[class_ind] = tf.multiply(class_scores, select_mask)

    return selected_bboxes, selected_scores

def clip_bboxes(ymin, xmin, ymax, xmax, name):
    with tf.name_scope(name, 'clip_bboxes', values=[ymin, xmin, ymax, xmax]):
        ymin = tf.maximum(ymin, 0.)
        xmin = tf.maximum(xmin, 0.)
        ymax = tf.minimum(ymax, 1.)
        xmax = tf.minimum(xmax, 1.)

        ymin = tf.minimum(ymin, ymax)
        xmin = tf.minimum(xmin, xmax)

        return ymin, xmin, ymax, xmax

def filter_bboxes(scores_pred, ymin, xmin, ymax, xmax, min_size, name):
    with tf.name_scope(name, 'filter_bboxes', values=[scores_pred, ymin, xmin, ymax, xmax]):
        width = xmax - xmin
        height = ymax - ymin

        filter_mask = tf.logical_and(width > min_size, height > min_size)

        filter_mask = tf.cast(filter_mask, tf.float32)
        return tf.multiply(ymin, filter_mask), tf.multiply(xmin, filter_mask), \
                tf.multiply(ymax, filter_mask), tf.multiply(xmax, filter_mask), tf.multiply(scores_pred, filter_mask)

def sort_bboxes(scores_pred, ymin, xmin, ymax, xmax, keep_topk, name):
    with tf.name_scope(name, 'sort_bboxes', values=[scores_pred, ymin, xmin, ymax, xmax]):
        cur_bboxes = tf.shape(scores_pred)[0]
        scores, idxes = tf.nn.top_k(scores_pred, k=tf.minimum(keep_topk, cur_bboxes), sorted=True)

        ymin, xmin, ymax, xmax = tf.gather(ymin, idxes), tf.gather(xmin, idxes), tf.gather(ymax, idxes), tf.gather(xmax, idxes)

        paddings_scores = tf.expand_dims(tf.stack([0, tf.maximum(keep_topk-cur_bboxes, 0)], axis=0), axis=0)

        return tf.pad(ymin, paddings_scores, "CONSTANT"), tf.pad(xmin, paddings_scores, "CONSTANT"),\
                tf.pad(ymax, paddings_scores, "CONSTANT"), tf.pad(xmax, paddings_scores, "CONSTANT"),\
                tf.pad(scores, paddings_scores, "CONSTANT")

def nms_bboxes(scores_pred, bboxes_pred, nms_topk, nms_threshold, name):
    with tf.name_scope(name, 'nms_bboxes', values=[scores_pred, bboxes_pred]):
        idxes = tf.image.non_max_suppression(bboxes_pred, scores_pred, nms_topk, nms_threshold)
        return tf.gather(scores_pred, idxes), tf.gather(bboxes_pred, idxes)

def parse_by_class(cls_pred, bboxes_pred, num_classes, select_threshold, min_size, keep_topk, nms_topk, nms_threshold):
    with tf.name_scope('select_bboxes', values=[cls_pred, bboxes_pred]):
        scores_pred = tf.nn.softmax(cls_pred)
        selected_bboxes, selected_scores = select_bboxes(scores_pred, bboxes_pred, num_classes, select_threshold)
        for class_ind in range(1, num_classes):
            ymin, xmin, ymax, xmax = tf.unstack(selected_bboxes[class_ind], 4, axis=-1)
            #ymin, xmin, ymax, xmax = tf.squeeze(ymin), tf.squeeze(xmin), tf.squeeze(ymax), tf.squeeze(xmax)
            ymin, xmin, ymax, xmax = clip_bboxes(ymin, xmin, ymax, xmax, 'clip_bboxes_{}'.format(class_ind))
            ymin, xmin, ymax, xmax, selected_scores[class_ind] = filter_bboxes(selected_scores[class_ind],
                                                ymin, xmin, ymax, xmax, min_size, 'filter_bboxes_{}'.format(class_ind))
            ymin, xmin, ymax, xmax, selected_scores[class_ind] = sort_bboxes(selected_scores[class_ind],
                                                ymin, xmin, ymax, xmax, keep_topk, 'sort_bboxes_{}'.format(class_ind))
            selected_bboxes[class_ind] = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
            selected_scores[class_ind], selected_bboxes[class_ind] = nms_bboxes(selected_scores[class_ind], selected_bboxes[class_ind], nms_topk, nms_threshold, 'nms_bboxes_{}'.format(class_ind))

        return selected_bboxes, selected_scores

def main(_):
    with tf.Graph().as_default():
        out_shape = [FLAGS.train_image_size] * 2

        image_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
        shape_input = tf.placeholder(tf.int32, shape=(2,))

        features = ssd_preprocessing.preprocess_for_eval(image_input, out_shape, data_format=FLAGS.data_format, output_rgb=False)
        features = tf.expand_dims(features, axis=0)

        anchor_creator = anchor_manipulator.AnchorCreator(out_shape,
                                                    #layers_shapes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
                                                    layers_shapes=[(38, 38), (19, 19), (10, 10),(5,5),(3,3),(1,1)],
                                                    #anchor_scales = [(0.1,), (0.2,), (0.375,), (0.55,), (0.725,), (0.9,)],
                                                    anchor_scales=[(0.1,), (0.2,), (0.375,), (0.55,),(0.725,),(0.9,)],
                                                    extra_anchor_scales = [(0.1414,), (0.2739,), (0.4541,), (0.6315,),(0.8078,),(0.9836,)],
                                                    anchor_ratios = [(1., 2.,.5), (1., 2., 3., .5, 0.3333), (1., 2., 3., .5, 0.3333),(1., 2., 3., .5, 0.3333), (1., 2., .5), (1., 2., .5)],
                                                    #anchor_ratios = [(2., .5), (2., 3., .5, 0.3333), (2., 3., .5, 0.3333), (2., 3., .5, 0.3333), (2., .5), (2., .5)],
                                                    #layer_steps = [8, 16, 32, 64, 100, 300]
                                                    layer_steps = [8,16,32,64,100,300])
        all_anchors, all_num_anchors_depth, all_num_anchors_spatial = anchor_creator.get_all_anchors()

        anchor_encoder_decoder = anchor_manipulator.AnchorEncoder(allowed_borders = [1.0] * 6,
                                                            positive_threshold = None,
                                                            ignore_threshold = None,
                                                            prior_scaling=[0.1, 0.1, 0.2, 0.2])

        decode_fn = lambda pred : anchor_encoder_decoder.ext_decode_all_anchors(pred, all_anchors, all_num_anchors_depth, all_num_anchors_spatial)

        with tf.variable_scope(FLAGS.model_scope, default_name=None, values=[features], reuse=tf.AUTO_REUSE):
            backbone = ssd_net.VGG16Backbone(FLAGS.data_format)
            feature_layers = backbone.forward(features, training=False)
            location_pred, cls_pred = ssd_net.multibox_head(feature_layers, FLAGS.num_classes, all_num_anchors_depth, data_format=FLAGS.data_format)
            if FLAGS.data_format == 'channels_first':
                cls_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in cls_pred]
                location_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in location_pred]

            cls_pred = [tf.reshape(pred, [-1, FLAGS.num_classes]) for pred in cls_pred]
            location_pred = [tf.reshape(pred, [-1, 4]) for pred in location_pred]

            cls_pred = tf.concat(cls_pred, axis=0)
            location_pred = tf.concat(location_pred, axis=0)

        with tf.device('/gpu:0'):
            bboxes_pred = decode_fn(location_pred)
            bboxes_pred = tf.concat(bboxes_pred, axis=0)
            selected_bboxes, selected_scores = parse_by_class(cls_pred, bboxes_pred,
                                                            FLAGS.num_classes, FLAGS.select_threshold, FLAGS.min_size,
                                                            FLAGS.keep_topk, FLAGS.nms_topk, FLAGS.nms_threshold)

            labels_list = []
            scores_list = []
            bboxes_list = []
            for k, v in selected_scores.items():
                labels_list.append(tf.ones_like(v, tf.int32) * k)
                scores_list.append(v)
                bboxes_list.append(selected_bboxes[k])
            all_labels = tf.concat(labels_list, axis=0)
            all_scores = tf.concat(scores_list, axis=0)
            all_bboxes = tf.concat(bboxes_list, axis=0)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_memory_fraction)
        config = tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        #os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
        saver = tf.train.Saver()
        cap = cv2.VideoCapture("E:/video/sperm/sperm_13.mp4")
        with tf.Session(config=config) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            saver.restore(sess, get_checkpoint())
            """
            np_image = imread('F:/Sperms/JPEGImages/000002.jpg')
            labels_, scores_, bboxes_ = sess.run([all_labels, all_scores, all_bboxes],
                                                 feed_dict={image_input: np_image, shape_input: np_image.shape[:-1]})

            img_to_draw = draw_toolbox.bboxes_draw_on_img(np_image, labels_, scores_, bboxes_, thickness=2)
            imsave('F:/sperm/00002_out.jpg', img_to_draw)
            """

            nodes = []
            frames_nodes = []
            n = 0
            G = nx.DiGraph()
            length = []
            coordinates = []
            track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                            (0, 127, 255), (255, 0, 255), (255, 127, 255),
                            (127, 0, 255), (127, 0, 127)]
            frame_count = 1

            while (True):
                ret, np_image = cap.read()

                labels_, scores_, bboxes_ = sess.run([all_labels, all_scores, all_bboxes],
                                                     feed_dict={image_input: np_image,shape_input: np_image.shape[:-1]})
                img_to_draw,detection = draw_toolbox.bboxes_draw_on_img(np_image, labels_, scores_, bboxes_, thickness=2)
                """
                if (len(detection) > 0):
                    length.append(len(detection))
                    single_frame_nodes = []
                    for i in range(len(detection)):
                        coordinates.append(detection[i])
                        single_frame_nodes.append(i + n)
                        nodes.append(i + n)
                        G.add_node(i + n)
                    frames_nodes.append(single_frame_nodes)
                if (len(length) > 1):
                    if (n > (n - length[-2])):
                        for i in nodes[(n - length[-2]):n]:
                            for j in nodes[n:]:
                                weigh = np.sqrt(
                                    abs(((coordinates[i][0][0] - coordinates[j][0][0]) * (coordinates[i][0][0] -
                                                                                          coordinates[j][0][0])) + (
                                                    coordinates[i][1][0] - coordinates[j][1][0]) *
                                        (coordinates[i][1][0] - coordinates[j][1][0])))
                                # print(i,j,weigh)
                                if (weigh < 65):
                                    G.add_edge(i, j, weight=weigh)

                    if frame_count < 5:
                        cost = np.zeros(shape=(length[0], len(detection)))
                        for i in range(length[0]):
                            for j in nodes[n:]:
                                try:
                                    path = nx.dijkstra_path(G, source=i, target=j)
                                    distance = nx.dijkstra_path_length(G, source=i, target=j)
                                    cost[i, j - n] = distance  # + angle
                                except nx.NetworkXNoPath:
                                    print(str(i) + " " + "No path")
                                    cost[i, j - n] = 99999999999999
                        row_ind, col_ind = linear_sum_assignment(cost)
                        col_ind = col_ind + n
                        paths = []
                        for i in range(len(row_ind)):
                            try:
                                path = nx.dijkstra_path(G, source=row_ind[i], target=col_ind[i])
                                distance = nx.dijkstra_path_length(G, source=row_ind[i], target=col_ind[i])
                                paths.append(path)
                            except nx.NetworkXNoPath:
                                print(str(i) + " " + "No path")

                    else:
                        cost = np.zeros(shape=(length[-5], length[-1]))
                        for i in nodes[sum(length[:frame_count - 5]):sum(length[:frame_count - 4])]:
                            for j in nodes[n:]:
                                try:
                                    path = nx.dijkstra_path(G, source=i, target=j)
                                    distance = nx.dijkstra_path_length(G, source=i, target=j)
                                    cost[i - sum(length[:frame_count - 5]), j - n] = distance  # + angle
                                except nx.NetworkXNoPath:
                                    print(str(i) + " " + "No path")
                                    cost[i - sum(length[:frame_count - 5]), j - n] = 99999999999999
                        row_ind, col_ind = linear_sum_assignment(cost)
                        row_ind = row_ind + sum(length[:frame_count - 5])
                        col_ind = col_ind + n
                        paths = []
                        for i in range(len(row_ind)):
                            try:
                                path = nx.dijkstra_path(G, source=row_ind[i], target=col_ind[i])
                                distance = nx.dijkstra_path_length(G, source=row_ind[i], target=col_ind[i])
                                paths.append(path)
                            except nx.NetworkXNoPath:
                                print(str(i) + " " + "No path")

                    path_current = paths
                    # print(paths)
                    # print(self.path)
                    for i in range(len(path_current)):
                        for track in range(len(path_current[i]) - 1):
                            if track > 0:
                                color = i % 9
                                x1 = int(coordinates[path_current[i][track]][0][0])
                                y1 = int(coordinates[path_current[i][track]][1][0])
                                x2 = int(coordinates[path_current[i][track + 1]][0][0])
                                y2 = int(coordinates[path_current[i][track + 1]][1][0])
                                #cv2.line(img_to_draw, (x1, y1), (x2, y2), track_colors[color], 2)
                frame_count += 1
                n += len(detection)
                """
                cv2.imshow("detection",img_to_draw)
                k=cv2.waitKey(30) & 0xff
                if k==27:
                    break
            cap.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
