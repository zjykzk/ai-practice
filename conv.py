#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author : zenk
# 2017-03-29 11:01
import numpy as np


def element_wise_op(array, op):
    for i in np.nditer(array, op_flags=['readwrite']):
        i[...] = op(i)


def get_patch(input_array, i, j, kernel_width, kernel_height, stride):
    return input_array[[i * stride, i * stride + kernel_width - 1],
                       [j * stride, j * stride + kernel_height - 1]]


def conv(input_array, kernel_array, output_array, stride, bias):
    #channel_number = input_array.ndim
    output_width = output_array.shape[1]
    output_height = output_array.shape[0]
    kernel_width = kernel_array.shape[-1]
    kernel_height = kernel_array.shape[-2]
    for i in range(output_height):
        for j in range(output_width):
            output_array[i][j] = (
                get_patch(input_array, i, j, kernel_width, kernel_height,
                          stride) * kernel_array).sum() + bias


def padding(input_array, zp):
    if zp == 0:
        return input_array

    if input_array.ndim == 3:
        input_depth, input_height, input_width = input_array.shape
        padded_array = np.zeros((input_depth, input_height + 2 * zp,
                                 input_width + 2 * zp))
        padded_array[:, zp:zp + input_height, zp:
                     zp + input_width] = input_array
        return padded_array

    if input_array.ndim == 2:
        input_height, input_width = input_array.shape
        padded_array = np.zeros((input_height + 2 * zp, input_width + 2 * zp))
        padded_array[zp:zp + input_height, zp:zp + input_width] = input_array
        return padded_array


class ConvLayer(object):
    @staticmethod
    def calculate_output_size(input_size, filter_size, zero_padding, stride):
        return (input_size - filter_size + 2 * zero_padding) / stride + 1

    def forward(self, input_array):
        self.input_array = input_array
        self.padded_input_array = padding(input_array, self.zero_padding)
        for filter, output in zip(self.filters, self.output_array):
            conv(self.padded_input_array,
                 filter.get_weights(), output, self.stride, filter.get_bias())
        element_wise_op(self.output_array, self.activator.forward)

    def __init__(self, input_width, input_height, channel_number, filter_width,
                 filter_height, filter_number, zero_padding, stride, activator,
                 learning_rate):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_number = filter_number
        self.zero_padding = zero_padding
        self.stride = stride
        self.output_width = ConvLayer.calculate_output_size(
            input_width, filter_width, zero_padding, stride)
        self.output_height = ConvLayer.calculate_output_size(
            input_height, filter_height, zero_padding, stride)
        self.output_array = np.zeros((self.filter_number, self.output_height,
                                      self.output_width))
        self.filters = [
            Filter(filter_width, filter_height, self.channel_number)
        ] * filter_number
        self.activator = activator
        self.learning_rate = learning_rate

    def expand_sensitivity_map(self, sensitivity_array):
        depth = sensitivity_array.shape[0]

        expand_array = np.zeros((
            depth,
            self.input_height + 2 * self.zero_padding + 1 - self.filter_height,
            self.input_width + 2 * self.zero_padding + 1 - self.filter_width))

        for i in range(self.output_height):
            for j in range(self.output_width):
                expand_array[:i * self.stride, j *
                             self.stride] = sensitivity_array[:, i, j]

        return expand_array


class Filter(object):
    def __init__(self, width, height, depth):
        self.weights = np.random.uniform(-1e-4, 1e-4, (depth, height, width))
        self.bias = 0
        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad = 0

    def __repr__(self):
        return 'filter weights:\n%s\nbias:\n%s' % (repr(self.weights),
                                                   repr(self.bias))

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def update(self, learning_rate):
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad


class ReluActivator(object):
    def forward(self, weighted_input):
        return max(0, weighted_input)

    def backward(self, output):
        return 1 if output > 0 else 0
