# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import numpy as np
import os, sys, importlib
from cntk import Trainer, load_model
from cntk.device import cpu, set_default_device
from cntk.learner import sgd
from cntk.blocks import Placeholder, Constant
from cntk.layers import Dense
from cntk.learner import momentum_sgd, learning_rate_schedule, momentum_as_time_constant_schedule
from cntk.ops import input_variable, constant, parameter, cross_entropy_with_softmax, classification_error, times, combine
from cntk.ops import roipooling
from cntk.ops.functions import CloneMethod
from cntk.io import ReaderConfig, ImageDeserializer, CTFDeserializer, StreamConfiguration
from cntk.initializer import glorot_uniform
import PARAMETERS
locals().update(importlib.import_module("PARAMETERS").__dict__)

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", ".."))
from examples.common.nn import print_training_progress

TRAIN_MAP_FILENAME = 'train.txt'
TEST_MAP_FILENAME = 'test.txt'
ROIS_FILENAME_POSTFIX = '.rois.txt'
ROILABELS_FILENAME_POSTFIX = '.roilabels.txt'


#####################################################
#####################################################
# helpers for graph traversal
def dfs_walk(node, visitor, accum, visited):
    if node in visited:
        return
    visited.add(node)
    print("visiting %s"%node.name)
    if hasattr(node, 'root_function'):
        node = node.root_function
        for child in node.inputs:
            dfs_walk(child, visitor, accum, visited)
    elif hasattr(node, 'is_output') and node.is_output:
        dfs_walk(node.owner, visitor, accum, visited)

    if visitor(node):
        accum.append(node)

def visit(root_node, visitor):
    nodes = []
    dfs_walk(root_node, visitor, nodes, set())
    return nodes

def find_nodes_by_name(root_node, node_name):
    return visit(root_node, lambda x: x.name == node_name)


# Instantiates a composite minibatch source for reading images, roi coordinates and roi labels for training Fast R-CNN
# The minibatch source is configured using a hierarchical dictionary of key:value pairs
def create_mb_source(features_stream_name, rois_stream_name, labels_stream_name, image_height,
                     image_width, num_channels, num_classes, num_rois, data_path, data_set):
    rois_dim = 4 * num_rois
    label_dim = num_classes * num_rois

    path = os.path.normpath(os.path.join(abs_path, data_path))
    map_file = os.path.join(path, TRAIN_MAP_FILENAME)
    roi_file = os.path.join(path, data_set + ROIS_FILENAME_POSTFIX)
    label_file = os.path.join(path, data_set + ROILABELS_FILENAME_POSTFIX)

    if not os.path.exists(map_file) or not os.path.exists(roi_file) or not os.path.exists(label_file):
        raise RuntimeError("File '%s', '%s' or '%s' does not exist. Please run install_fastrcnn.py from Examples/Image/Detection/FastRCNN to fetch them" %
                           (map_file, roi_file, label_file))

    # read images
    # ??? do we still need 'transpose'?
    image_source = ImageDeserializer(map_file)
    image_source.ignore_labels()
    image_source.map_features(features_stream_name,
        [ImageDeserializer.scale(width=image_width, height=image_height, channels=num_channels,
                                 scaleMode="pad", padValue=114, interpolations='linear')])

    # read rois and labels
    roi_source = CTFDeserializer(roi_file)
    roi_source.map_input(rois_stream_name, dim=rois_dim, format="dense")
    label_source = CTFDeserializer(label_file)
    label_source.map_input(labels_stream_name, dim=label_dim, format="dense")

    # define a composite reader
    rc = ReaderConfig([image_source, roi_source, label_source], epoch_size=sys.maxsize)
    return rc.minibatch_source()


# Defines the Fast R-CNN network model for detecting objects in images
def frcn_predictor(features, rois, num_classes):
    # Load the pretrained model and find nodes
    loaded_model = load_model("../../../../../PretrainedModels/AlexNetBS.model", 'float')
    feature_node = find_nodes_by_name(loaded_model, "features")
    conv5_node   = find_nodes_by_name(loaded_model, "z.x._.x._.x.x_output")
    pool3_node   = find_nodes_by_name(loaded_model, "z.x._.x._.x_output")
    h2d_node     = find_nodes_by_name(loaded_model, "z.x_output")

    # Clone the conv layers of the network, i.e. from the input features up to the output of the 5th conv layer
    conv_layers = combine([conv5_node[0].owner]).clone(CloneMethod.freeze, {feature_node[0]: Placeholder()})

    # Clone the fully connected layers, i.e. from the output of the last pooling layer to the output of the last dense layer
    fc_layers = combine([h2d_node[0].owner]).clone(CloneMethod.clone, {pool3_node[0]: Placeholder()})

    # create Fast R-CNN model
    feat_norm = features - Constant(114)
    conv_out  = conv_layers(feat_norm)
    roi_out   = roipooling(conv_out, rois, (6,6)) # rename to roi_max_pooling
    fc_out    = fc_layers(roi_out)

    # z = Dense((rois[0], num_classes), map_rank=1)(fc_out) --> map_rank=1 is not yet supported
    W = parameter(shape=(4096, num_classes), init=glorot_uniform())
    b = parameter(shape=(num_classes), init=0)
    z = times(fc_out, W) + b
    return z


# Trains a Fast R-CNN network model on the grocery image dataset
def frcn_grocery(base_path, debug_output=False):
    num_channels = 3
    image_height = cntk_padHeight   # from PARAMETERS.py
    image_width = cntk_padWidth     # from PARAMETERS.py
    num_classes = nrClasses         # from PARAMETERS.py
    num_rois = cntk_nrRois          # from PARAMETERS.py
    feats_stream_name = 'features'
    rois_stream_name = 'rois'
    labels_stream_name = 'roiLabels'

    #####
    # training
    minibatch_source = create_mb_source(feats_stream_name, rois_stream_name, labels_stream_name,
                       image_height, image_width, num_channels, num_classes, num_rois, base_path, "train")
    features_si = minibatch_source[feats_stream_name]
    rois_si = minibatch_source[rois_stream_name]
    labels_si = minibatch_source[labels_stream_name]

    # Input variables denoting features, rois and label data
    image_input = input_variable((num_channels, image_height, image_width), features_si.m_element_type)
    roi_input = input_variable((num_rois, 4), rois_si.m_element_type)
    label_input = input_variable((num_rois, num_classes), labels_si.m_element_type)

    # Instantiate the Fast R-CNN prediction model
    frcn_output = frcn_predictor(image_input, roi_input, num_classes)

    ce = cross_entropy_with_softmax(frcn_output, label_input, axis=1)
    pe = classification_error(frcn_output, label_input, axis=1)

    # Set learning parameters
    epoch_size = 25                    # for now we manually specify epoch size
    mb_size = 1
    max_epochs = 17
    momentum_time_constant = -mb_size/np.log(0.9)
    l2_reg_weight = 0.0005

    lr_per_mb = [0.00001] * 10 + [0.000001] * 5 + [0.0000001]
    lr_per_sample = [lr / mb_size for lr in lr_per_mb]
    lr_schedule = learning_rate_schedule(lr_per_sample, epoch_size=epoch_size)
    mm_schedule = momentum_as_time_constant_schedule(momentum_time_constant)

    # Instantiate the trainer object to drive the model training
    learner = momentum_sgd(frcn_output.parameters, lr_schedule, mm_schedule, l2_regularization_weight=l2_reg_weight)
    trainer = Trainer(frcn_output, ce, pe, learner)

    # Get minibatches of images to train with and perform model training
    training_progress_output_freq = int(epoch_size / mb_size)
    num_mbs = int(epoch_size * max_epochs / mb_size) # 17 epochs * 25 images / 1 mbSize

    if debug_output:
        training_progress_output_freq = training_progress_output_freq / 10

    # Main training loop
    for i in range(0, num_mbs):
        mb = minibatch_source.next_minibatch(mb_size)

        # Specify the mapping of input variables in the model to actual minibatch data to be trained with
        arguments = {
                image_input: mb[features_si],
                roi_input: mb[rois_si],
                label_input: mb[labels_si]
                }
        trainer.train_minibatch(arguments)
        print_training_progress(trainer, i, training_progress_output_freq)

    #####
    # testing
    test_minibatch_source = create_mb_source(feats_stream_name, rois_stream_name, labels_stream_name,
                    image_height, image_width, num_channels, num_classes, num_rois, base_path, "test")
    features_si = test_minibatch_source[feats_stream_name]
    rois_si     = test_minibatch_source[rois_stream_name]

    mb_size = 1
    num_mbs = 5

    results_file_path = base_path + "test.z"
    with open(results_file_path, 'wb') as results_file:
        for i in range(0, num_mbs):
            mb = test_minibatch_source.next_minibatch(mb_size)

            # Specify the mapping of input variables in the model to actual minibatch data to be tested with
            arguments = {
                    image_input: mb[features_si],
                    roi_input:   mb[rois_si],
                    }
            output = trainer.model.eval(arguments)
            out_values = output[0,0].flatten()
            np.savetxt(results_file, out_values[np.newaxis], fmt="%.6f")

    return

#if __name__ == '__main__':
# Specify the target device to be used for computing, if you do not want to
# use the best available one, e.g.
# set_default_device(cpu())

os.chdir(cntkFilesDir)
frcn_grocery(cntkFilesDir)
