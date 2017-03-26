#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author : zenk
# 2017-03-26 20:16

import gzip
import struct

import bp


class Loader(object):
    def __init__(self, path, count):
        self.path = path
        self.count = count

    def get_file_content(self):
        with gzip.open(self.path) as f:
            return f.read()

    def to_int(self, byte):
        return struct.unpack('B', byte)[0]


class ImageLoader(Loader):
    def get_picture(self, content, index):
        start = index * 28 * 28 + 16
        picture = []
        for i in range(28):
            picture.append(
                [self.to_int(content[start + i * 28 + j]) for j in range(28)])

        return picture

    def get_one_sample(self, picture):
        return [picture[i][j] for i in range(28) for j in range(28)]

    def load(self):
        content = self.get_file_content()
        return [
            self.get_one_sample(self.get_picture(content, i))
            for i in range(self.count)
        ]

    
class LabelLoader(Loader):
    def load(self):
        content = self.get_file_content()
        return [self.norm(content[i + 8]) for i in range(self.count)]

    def norm(self, label):
        label_value = self.to_int(label)
        return [.95 if i == label_value else .05 for i in range(10)]


def get_training_data_set():
    image_loader = ImageLoader('./train-images-idx3-ubyte.gz', 60000)
    label_loader = LabelLoader('./train-labels-idx1-ubyte.gz', 60000)

    return image_loader.load(), label_loader.load()

def get_test_data_set():
    image_loader = ImageLoader('./t10k-images-idx3-ubyte.gz', 10000)
    label_loader = LabelLoader('./t10k-labels-idx1-ubyte.gz', 10000)

    return image_loader.load(), label_loader.load()


def get_result(vec):
    max_value_index = 0
    max_value = 0
    for i, v in enumerate(vec):
        if v > max_value:
            max_value = v
            max_value_index = i

    return max_value_index


def evaluate(network, test_data_set, test_labels):
    correct, total = 0, len(test_data_set)

    for i in range(total):
        label = get_result(test_labels[i])
        predict = get_result(network.predict(test_data_set[i]))
        if label == predict:
            correct += 1

    return float(correct) /float(total)

def train_and_evaluate():
    last_error_ratio = 1.0
    epoch = 0
    train_data_set, train_labels = get_training_data_set()
    test_data_set, test_labels = get_test_data_set()

    network = bp.Network([784, 300, 10])

    while True:
        epoch += 10
        network.train(train_data_set, train_labels, .1, 10)
        error_ratio = evaluate(network, test_data_set, test_labels)
        print('after epoch %d, error ratio is %f' % (epoch, error_ratio))
        if error_ratio > last_error_ratio:
            break

        last_error_ratio = error_ratio


if __name__ == '__main__':
    train_and_evaluate()
