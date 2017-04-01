#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author : zenk
# 2017-03-29 11:01
import numpy as np


def element_wise_op(array, op):
    for i in np.nditer(array, op_flags=['readwrite']):
        i[...] = op(i)


def get_patch(input_array, i, j, kernel_width, kernel_height, stride):
    return input_array[i * stride:i * stride + kernel_width, j * stride:
                       j * stride + kernel_height]


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

    def bp_sensitivity_map(self, sensitivity_array, activator):
        expanded_array = self.expand_sensitivity_map(sensitivity_array)
        expanded_width = expanded_array.shape[2]
        zp = (self.input_width + self.filter_width - 1 - expanded_width) / 2
        padded_array = padding(expanded_array, zp)
        self.delta_array = self.create_delta_array()
        for f in range(self.filter_number):
            filter = self.filters[f]
            filpped_weights = np.array(
                map(lambda i: np.rot90(i, 2), filter.get_weights()))
            delta_array = self.create_delta_array()
            for d in range(delta_array.shape[0]):
                conv(padded_array[f], filpped_weights[d], delta_array[d], 1, 0)
            self.delta_array += delta_array

        derivative_array = np.array(self.input_array)
        element_wise_op(derivative_array, activator.backward)
        self.delta_array *= derivative_array

    def create_delta_array(self):
        return np.zeros(self.channel_number, self.input_height,
                        self.input_width)

    def bp_gradient(self, sensitivity_array):
        expanded_array = self.expand_sensitivity_map(sensitivity_array)
        for f in range(self.filter_number):
            filter = self.filter[f]
            for d in range(filter.weights.shape[0]):
                conv(self.padded_input_array[d], expanded_array[f],
                     filter.weights_grad[d], 1, 0)
        filter.bias_grad = expanded_array[f].sum()

    def update(self):
        for filter in self.filters:
            filter.update(self.learning_rate)

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


class IdentityActivator(object):
    def forward(self, weighted_input):
        return weighted_input

    def backward(self, output):
        return 1


def init_test():
    a = np.array(
        [[[0, 1, 1, 0, 2], [2, 2, 2, 2, 1], [1, 0, 0, 2, 0], [0, 1, 1, 0, 0],
          [1, 2, 0, 0, 2]], [[1, 0, 2, 2, 0], [0, 0, 0, 2, 0], [1, 2, 1, 2, 1],
                             [1, 0, 0, 0, 0], [1, 2, 1, 1, 1]],
         [[2, 1, 2, 0, 0], [1, 0, 0, 1, 0], [0, 2, 1, 0, 1], [0, 1, 2, 2, 2],
          [2, 1, 0, 0, 1]]])

    b = np.array([[[0, 1, 1], [2, 2, 2], [1, 0, 0]], [[1, 0, 2], [0, 0, 0],
                                                      [1, 2, 1]]])
    c1 = ConvLayer(5, 5, 3, 3, 3, 2, 1, 2, IdentityActivator(), 0.001)
    c1.filters[0].weights = np.array(
        [[[-1, 1, 0], [0, 1, 1], [0, 1, 1]],
         [[-1, -1, 0], [0, 0, 0], [0, -1, 0]], [[0, 0, -1], [0, 1, 0],
                                                [1, -1, -1]]],
        dtype=np.float64)
    c1.filters[0].bias = 1
    c1.filters[1].weights = np.array(
        [[[1, 1, -1], [-1, -1, 1], [0, -1, 1]],
         [[0, 1, 0], [-1, 0, -1], [0, -1, 1]], [[-1, 0, 0], [-1, 0, 1],
                                                [-1, 0, 0]]],
        dtype=np.float64)

    return a, b, c1


def gradient_check():
    error_function = lambda o: o.sum()

    a, b, c1 = init_test()
    sensitivity_array = np.ones(c1.output_array.shape, dtype=np.float64)

    c1.backward(a, sensitivity_array, IdentityActivator())
    epsilion = 10e-4
    for d in range(c1.filters[0].weights_grad.shape[2]):
        for i in range(c1.filters[0].weights_grad.shape[1]):
            for j in range(c1.filters[0].weights_grad.shape[0]):
                c1.filters[0].weights[d, i, j] += epsilion
                c1.forward(a)
                err1 = error_function(c1.output_array)
                c1.filters[0].weights[d, i, j] -= 2 * epsilion
                c1.forward(a)
                err2 = error_function(c1.output_array)
                expect_grad = (err1 - err2) / (2 * epsilion)
                c1.filters[0].weights[d, i, j] += epsilion
                print('weights(%d,%d,%d): expected - actual %f - %f' %
                    (d, i, j, expect_grad, c1.filters[0].weights_grad[d, i, j]))


if __name__ == '__main__':
    gradient_check()
