# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

from __future__ import print_function
import importlib
from cntk import *
from cntk.blocks import Placeholder, Constant
from cntk.learner import momentum_sgd, learning_rate_schedule, momentum_as_time_constant_schedule
from cntk.ops import input_variable, parameter, cross_entropy_with_softmax, classification_error, times, combine
from cntk.ops import roipooling
from cntk.ops.functions import CloneMethod
from cntk.io import ReaderConfig, ImageDeserializer, CTFDeserializer, StreamConfiguration
from cntk.initializer import glorot_uniform
from cntk.graph import find_by_name, plot
locals().update(importlib.import_module("PARAMETERS").__dict__)

###############################################################
###############################################################
abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, "..", ".."))

# file and stream names
train_map_filename = 'train.txt'
test_map_filename = 'test.txt'
rois_filename_postfix = '.rois.txt'
roilabels_filename_postfix = '.roilabels.txt'
features_stream_name = 'features'
rois_stream_name = 'rois'
labels_stream_name = 'roiLabels'

# from PARAMETERS.py
base_path = cntkFilesDir
num_channels = 3
image_height = cntk_padHeight
image_width = cntk_padWidth
num_classes = nrClasses
num_rois = cntk_nrRois
epoch_size = cntk_num_train_images
num_test_images = cntk_num_test_images
mb_size = cntk_mb_size
max_epochs = cntk_max_epochs
momentum_time_constant = cntk_momentum_time_constant

# model specific variables (only AlexNet for now)
base_model = "AlexNet"
if (base_model == "AlexNet"):
    model_file = "../../../../../PretrainedModels/AlexNet.model"
    feature_node_name = "features"
    last_conv_node_name = "conv5.y"
    pool_node_name = "pool3"
    last_hidden_node_name = "h2_d"
    roi_dim = 6
else:
    raise ValueError('unknown base model: %s' % base_model)
###############################################################
###############################################################

# TODO: use cntk functions
def print_training_progress(trainer, mb, frequency):
    if mb % frequency == 0:
        training_loss = get_train_loss(trainer)
        eval_crit = get_train_eval_criterion(trainer)
        print("Minibatch: {}, Train Loss: {}, Train Evaluation Criterion: {}".format(
            mb, training_loss, eval_crit))


# Instantiates a composite minibatch source for reading images, roi coordinates and roi labels for training Fast R-CNN
def create_mb_source(img_height, img_width, img_channels, n_classes, n_rois, data_path, data_set):
    rois_dim = 4 * n_rois
    label_dim = n_classes * n_rois

    path = os.path.normpath(os.path.join(abs_path, data_path))
    if (data_set == 'test'):
        map_file = os.path.join(path, test_map_filename)
    else:
        map_file = os.path.join(path, train_map_filename)
    roi_file = os.path.join(path, data_set + rois_filename_postfix)
    label_file = os.path.join(path, data_set + roilabels_filename_postfix)

    if not os.path.exists(map_file) or not os.path.exists(roi_file) or not os.path.exists(label_file):
        raise RuntimeError("File '%s', '%s' or '%s' does not exist. Please run install_fastrcnn.py from Examples/Image/Detection/FastRCNN to fetch them" %
                           (map_file, roi_file, label_file))

    # read images
    image_source = ImageDeserializer(map_file)
    image_source.ignore_labels()
    image_source.map_features(features_stream_name,
        [ImageDeserializer.scale(width=img_width, height=img_height, channels=img_channels,
                                 scale_mode="pad", pad_value=114, interpolations='linear')])

    # read rois and labels
    roi_source = CTFDeserializer(roi_file)
    roi_source.map_input(rois_stream_name, dim=rois_dim, format="dense")
    label_source = CTFDeserializer(label_file)
    label_source.map_input(labels_stream_name, dim=label_dim, format="dense")

    # define a composite reader
    rc = ReaderConfig([image_source, roi_source, label_source], epoch_size=sys.maxsize, randomize=data_set=="train")
    return rc.minibatch_source()


# Defines the Fast R-CNN network model for detecting objects in images
def frcn_predictor(features, rois, num_classes, plot_graphs=False):
    # Load the pretrained model and find nodes
    loaded_model = load_model(model_file)
    feature_node = find_by_name(loaded_model, feature_node_name)
    conv_node    = find_by_name(loaded_model, last_conv_node_name)
    pool_node    = find_by_name(loaded_model, pool_node_name)
    last_node    = find_by_name(loaded_model, last_hidden_node_name)

    # Clone the conv layers of the network, i.e. from the input features up to the output of the last conv layer
    conv_layers = combine([conv_node.owner]).clone(CloneMethod.freeze, {feature_node: Placeholder()})
    if plot_graphs: plot(conv_layers, os.path.join(abs_path, "Output", "graph_conv.png"))

    # Clone the fully connected layers, i.e. from the output of the last pooling layer to the output of the last dense layer
    fc_layers = combine([last_node.owner]).clone(CloneMethod.clone, {pool_node: Placeholder()})
    if plot_graphs: plot(fc_layers, os.path.join(abs_path, "Output", "graph_fc.png"))

    # create Fast R-CNN model
    feat_norm = features - Constant(114)
    conv_out  = conv_layers(feat_norm)
    roi_out   = roipooling(conv_out, rois, (roi_dim,roi_dim)) # rename to roi_max_pooling
    fc_out    = fc_layers(roi_out)

    # z = Dense((rois[0], num_classes), map_rank=1)(fc_out) --> map_rank=1 is not yet supported
    W = parameter(shape=(4096, num_classes), init=glorot_uniform())
    b = parameter(shape=(num_classes), init=0)
    z = times(fc_out, W) + b

    if plot_graphs: plot(z, os.path.join(abs_path, "Output", "graph_frcn.png"))
    return z


# Trains a Fast R-CNN network model
def train_fast_rcnn():
    # Create the minibatch source
    minibatch_source = create_mb_source(image_height, image_width, num_channels, num_classes, num_rois, base_path, "train")
    features_si = minibatch_source[features_stream_name]
    rois_si     = minibatch_source[rois_stream_name]
    labels_si   = minibatch_source[labels_stream_name]

    # Input variables denoting features, rois and label data
    image_input = input_variable((num_channels, image_height, image_width), features_si.m_element_type)
    roi_input   = input_variable((num_rois, 4), rois_si.m_element_type)
    label_input = input_variable((num_rois, num_classes), labels_si.m_element_type)

    # Instantiate the Fast R-CNN prediction model and loss function
    frcn_output = frcn_predictor(image_input, roi_input, num_classes)
    ce = cross_entropy_with_softmax(frcn_output, label_input, axis=1)
    pe = classification_error(frcn_output, label_input, axis=1)

    # Set learning parameters
    l2_reg_weight = 0.0005
    lr_per_sample = [0.00001] * 10 + [0.000001] * 5 + [0.0000001]
    lr_schedule = learning_rate_schedule(lr_per_sample, unit=UnitType.sample)
    mm_schedule = momentum_as_time_constant_schedule(momentum_time_constant)

    # Instantiate the trainer object to drive the model training
    learner = momentum_sgd(frcn_output.parameters, lr_schedule, mm_schedule, l2_regularization_weight=l2_reg_weight)
    trainer = Trainer(frcn_output, ce, pe, learner)

    # Get minibatches of images to train with and perform model training
    training_progress_output_freq = np.min([100, int(epoch_size / mb_size)])
    num_mbs = 5 # int(epoch_size * max_epochs / mb_size)
    print ("Training Fast R-CNN model for %s mini batches." % num_mbs)

    # Main training loop
    for i in range(0, num_mbs):
        mb = minibatch_source.next_minibatch(mb_size)
        arguments = {
            image_input: mb[features_si],
            roi_input: mb[rois_si],
            label_input: mb[labels_si]
        }
        trainer.train_minibatch(arguments)
        print_training_progress(trainer, i, training_progress_output_freq)

    trainer.save_checkpoint(os.path.join(abs_path, "Output", "frcn_model_py.model"))

#    return trainer.model

# Tests a Fast R-CNN network model
#def test_fast_rcnn(model):
    test_minibatch_source = create_mb_source(image_height, image_width, num_channels, num_classes, num_rois, base_path, "test")
    features_si = test_minibatch_source[features_stream_name]
    rois_si     = test_minibatch_source[rois_stream_name]
    #image_input = input_variable((num_channels, image_height, image_width), features_si.m_element_type)
    #roi_input   = input_variable((num_rois, 4), rois_si.m_element_type)

    results_file_path = base_path + "test.z"
    with open(results_file_path, 'wb') as results_file:
        for i in range(0, num_test_images):
            mb = test_minibatch_source.next_minibatch(1)

            # Specify the mapping of input variables in the model to actual minibatch data to be tested with
            arguments = {
                    image_input: mb[features_si],
                    roi_input:   mb[rois_si],
                    }
            output = trainer.model.eval(arguments)
            out_values = output[0,0].flatten()
            np.savetxt(results_file, out_values[np.newaxis], fmt="%.6f")
            if(i % 100 == 0):
                print("Evaluated %s images.." % i)

    return

#if __name__ == '__main__':
# Specify the target device to be used for computing, if you do not want to
# use the best available one, e.g.
# set_default_device(cpu())

os.chdir(base_path)
trained_model = train_fast_rcnn()
#test_fast_rcnn(trained_model)

# Generate ROIs (separate)
# Train Fast R-CNN model, store checkpoints and model
# Eval Test set (from stored model)
# Eval single image (separate, from stored model)

