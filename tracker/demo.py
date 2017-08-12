#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen Huang, based on code from Ross Girshick
# --------------------------------------------------------

'''
Demo script showing detections in sample images.

See README.md for installation instructions before running.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.test import im_detect
from model.nms_wrapper import nms

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
import deep_sort.feature_extractor as feature_extractor

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def gather_sequence_info(sequence_dir):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:

        * sequence_name: Name of the sequence
        * image_filenames: A dictionary that maps frame indices to image
          filenames.
        * detections: A numpy array of detections in MOTChallenge format.
        * groundtruth: A numpy array of ground truth in MOTChallenge format.
        * image_size: Image size (height, width).
        * min_frame_idx: Index of the first frame.
        * max_frame_idx: Index of the last frame.

    """
    # image_dir = os.path.join(sequence_dir, "img1")
    image_dir = sequence_dir
    image_filenames = {
            int(f[6:10]): os.path.join(image_dir, f) 
            for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))}
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    detections = None
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys())
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = 0
        max_frame_idx = 0

    info_filename = os.path.join(sequence_dir, "seqinfo.ini")
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(
                s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


def create_detections(detection_mat, min_height=0):
    """Create detections for given frame index from the raw detection matrix.

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    detection_list = []
    for row in detection_mat:
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list

def run(detect_net, track_net, dataset, sequence_dir, output_file, 
        min_confidence, nms_max_overlap, min_detection_height, 
        max_cosine_distance, nn_budget, display):
    """Run multi-target tracker on a particular sequence.
 
    Parameters
    ----------
    detect_net: str
        Network for detection
    track_net: str
        Path to tracking network
    dataset: str
        Dataset used to train detect_net
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    seq_info = gather_sequence_info(sequence_dir)
    metric = nn_matching.NearestNeighborDistanceMetric(
        'cosine', max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    results = []

    # load object detector
    detect_model = os.path.join('output', detect_net, DATASETS[dataset][0], 'default',
                                NETS[detect_net][0])

    if not os.path.isfile(detect_model + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(detect_model + '.meta'))
    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    detection_graph = tf.Graph()
    sess = tf.Session(graph=detection_graph, config=tfconfig)
    with detection_graph.as_default():
        # load network
        if detect_net == 'vgg16':
            net = vgg16(batch_size=1)
        elif detect_net == 'res101':
            net = resnetv1(batch_size=1, num_layers=101)
        else:
            raise NotImplementedError
        net.create_architecture("TEST", 21,
                                tag='default', anchor_scales=[8, 16, 32])
        saver = tf.train.Saver()
        saver.restore(sess, detect_model)
        print('Loaded network {:s}'.format(detect_model))

    encoder = feature_extractor.create_box_encoder(track_net)

    def frame_callback(vis, frame_idx):
        image_np = cv2.imread(seq_info['image_filenames'][frame_idx], cv2.IMREAD_COLOR)
        scores, boxes = im_detect(sess, net, image_np)

        # extract boxes for people
        cls = 'person'
        cls_ind = CLASSES.index(cls)
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, nms_max_overlap)
        keep = [ind for ind in keep if cls_scores[ind] > min_confidence]
        num_boxes = len(keep)

        # create rows based on MOTChallange format
        frame_ids = np.full(num_boxes, frame_idx)
        person_ids = np.full(num_boxes, -1)
        extra_cols = np.full((num_boxes, 3), -1)
        cls_boxes[:,2:] -= cls_boxes[:,:2]
        dets = np.hstack((frame_ids[:, np.newaxis], 
                          person_ids[:, np.newaxis],
                          cls_boxes[keep, :],
                          cls_scores[keep, np.newaxis],
                          extra_cols)).astype(np.float32)

        features = encoder(image_np, dets[:, 2:6].copy())
        detections_out = [np.r_[(dets, feature)] for dets, feature
                           in zip(dets, features)]
        detections = create_detections(detections_out, min_detection_height)

        # Update tracker.
        tracker.predict()
        tracker.update(detections)

        # Update visualization
        if display:
            vis.set_image(image_np.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)

        # Store results
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
        
    # Run tracker
    if display:
        visualizer = visualization.Visualization(seq_info, update_ms=50)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    visualizer.run(frame_callback)

    # Store results
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]), file=f)
    



def parse_args():
    '''Parse input arguments.'''
    parser = argparse.ArgumentParser(description='Deep SORT')
    parser.add_argument('--detect_net', dest='detect_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    parser.add_argument('--track_net', dest='track_net', help='Network to use for feature extraction',
                        default="output/mars/mars-small128.ckpt-68577")
    parser.add_argument('--sequence_dir', help='Path to sequence directory',
                        default=None, required=True)
    parser.add_argument('--output_file', help='Path to the tracking output file. This file will'
                        'contain the tracking results on completion.',
                        default='/tmp/hypotheses.txt')
    parser.add_argument('--min_confidence', help='Detection confidence threshold. Disregard '
                        'all detections that have a confidence lower than this value.',
                        default=0.8, type=float)
    parser.add_argument('--min_detection_height', help='Threshold on the detection bounding '
                        'box height. Detections with height smaller than this value are '
                        'disregarded', default=0, type=int)
    parser.add_argument('--nms_max_overlap',  help='Non-maxima suppression threshold: Maximum '
                        'detection overlap.', default=0.3, type=float)
    parser.add_argument('--max_cosine_distance', help='Gating threshold for cosine distance '
                        'metric (object appearance).', type=float, default=0.2)
    parser.add_argument('--nn_budget', help='Maximum size of the appearance descriptors '
                        'gallery. If None, no budget is enforced.', type=int, default=None)
    parser.add_argument('--display', help='Show intermediate tracking results',
                        default=True, type=bool)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    run(args.detect_net, args.track_net, args.dataset, args.sequence_dir, 
        args.output_file, args.min_confidence, args.nms_max_overlap, 
        args.min_detection_height, args.max_cosine_distance, args.nn_budget, args.display)