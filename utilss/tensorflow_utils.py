import tensorflow as tf
from tensorflow.contrib.model_pruning.python import pruning
import numpy as np
import matplotlib.pyplot as plt


def _build_conv_layer(self, inputs, scope, weight_init, filter_height,
                      filter_width, channel_in, channel_out, strides, padding='SAME', dataformat="NHWC"):
    kernel = self._variable_with_weight_decay(
        'weights', shape=[filter_height, filter_width, channel_in, channel_out], initialization=weight_init, wd=0.0)
    conv = tf.nn.conv2d(input=inputs, filter=pruning.apply_mask(kernel, scope),
                        padding=padding, strides=strides, data_format=dataformat)
    biases = self._variable_on_cpu('biases', channel_out, initializer=tf.constant_initializer(0.001))
    pre_activation = tf.nn.bias_add(conv, biases, data_format=dataformat)
    return pre_activation


def _build_fc_layer(self, inputs, scope, weight_init, shape, activation=None):
    weights = self._variable_with_weight_decay(
        'weights', shape=shape, initialization=weight_init, wd=0.0)
    biases = self._variable_on_cpu('biases', shape[1], initializer=tf.constant_initializer(0.001))
    if activation is not None:
        return activation(
            tf.matmul(inputs, pruning.apply_mask(weights, scope)) + biases,
            name=scope.name)
    else:
        return tf.matmul(inputs, pruning.apply_mask(weights, scope)) + biases


def imshow_noax(img, normalize=False):
    """ Tiny helper to show images as uint8 and remove axis labels """
    if normalize:
        img_max, img_min = np.max(img), np.min(img)
        img = 55.0 * (img - img_min) / (img_max - img_min)
    plt.imshow(img.astype('uint8'))
    plt.gca().axis('off')
    plt.show()


def calculate_redundancy(initial_nnz_params, next_nnz_params):
    redundancy_precenct = []
    for i, nnz_parms in enumerate(next_nnz_params):
        redundancy_precenct.append(1 - (nnz_parms / initial_nnz_params[i]))
    #redundancy_precenct[3] = (redundancy_precenct[3] + redundancy_precenct[4]) / 2
    # one size effect two matrices, because the input size of fc_3 is fixed by the size of filters
    return redundancy_precenct


def transform_one_hot(index, size):
    assert index < size
    return [0 if i != index else 1 for i in range(size)]


def transform_one_hot_batch(batch_index, size):
    return [transform_one_hot(index, size) for i, index in enumerate(batch_index)]


def reshape_input_for_critic(inputs):
    assert np.shape(inputs)[0] > 1,\
        "maddpg work with multiple agents so states should be in an array with " \
        "the first axis representing the number of agents in the system"
    critic_input = inputs[0]
    for inputs_i in inputs[1:]:
        critic_input = np.concatenate((critic_input, inputs_i), axis=-1)
    return critic_input