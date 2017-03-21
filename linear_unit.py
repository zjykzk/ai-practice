#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author : zenk
# 2017-03-21 21:36
from perceptron import Perceptron

class LinearUnit(Perceptron):
    def __init__(self, input_num):
        Perceptron.__init__(self, input_num, lambda x: x)


def get_trainning_dataset():
    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    labels = [5500, 2300, 7600, 1800, 11400]

    return input_vecs, labels


def train_linear_unit():
    lu = LinearUnit(1)
    input_vecs, labels = get_trainning_dataset()
    lu.train(input_vecs, labels, 0.1, 10)
    return lu


if __name__ == '__main__':
    linear_unit = train_linear_unit()
    print linear_unit

    for y in [[3.4], [15], [1.5], [6.3]]:
        print('work %.2f years, monthly salary = %.2f' % (y[0], linear_unit.predict(y)))
