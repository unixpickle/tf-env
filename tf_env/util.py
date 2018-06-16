"""
Utilities for making batched environments.
"""

import tensorflow as tf


def bcast_where(cond, tensor_1, tensor_2):
    """
    A where() that broadcasts the two options to be of the
    same shape. This is useful for selecting between a
    vector and a constant.
    """
    if len(tensor_1.get_shape()) < len(tensor_2.get_shape()):
        tensor_1 = tensor_1 + tf.zeros_like(tensor_2)
    elif len(tensor_1.get_shape()) > len(tensor_2.get_shape()):
        tensor_2 = tensor_2 + tf.zeros_like(tensor_1)
    return tf.where(cond, tensor_1, tensor_2)
