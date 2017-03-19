#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author : zenk
# 2017-03-17 13:56
class Perceptron(object):
    def __init__(self, input_num, activator):
        self.activator = activator
        self.weights = [0.0] * input_num
        self.bias = 0.0

    def __str__(self):
        return 'weights:%s\tbias:%f\n' % (self.weights, self.bias)

    def predict(self, input_vec):
        return self.activator(reduce(lambda a, b: a + b,
                                     map(lambda (a, b): a * b,
                                         zip(self.weights, input_vec)),
                                     0.0) + self.bias)

    def _update_weights(self, input_vec, output, label, rate):
        delta = label - output
        self.weights = map(lambda (x, w): w + rate * delta * x,
                           zip(input_vec, self.weights))
        self.bias += rate * delta

    def _one_iter(self, input_vecs, labels, rate):
        for input_vec, label in zip(input_vecs, labels):
            self._update_weights(input_vec,
                                 self.predict(input_vec),
                                 label,
                                 rate)

    def train(self, input_vecs, labels, rate, iter_count):
        for _ in range(iter_count):
            self._one_iter(input_vecs, labels, rate)


def f(x):
    return 1 if x > 0 else 0


def get_trainning_dataset():
    input_vecs = [[1,1],[1,0],[0,1],[0,0]]
    labels = [1, 0, 0, 0]

    return input_vecs, labels


def tain_and_perceptron():
    p = Perceptron(2, f)
    input_vecs, labels = get_trainning_dataset()
    p.train(input_vecs, labels, .10, 10)
    return p


if __name__ == '__main__':
    p = tain_and_perceptron()
    print p

    print '1 and 1 = %d' % p.predict([1, 1])
    print '1 and 0 = %d' % p.predict([1, 0])
    print '0 and 1 = %d' % p.predict([0, 1])
    print '0 and 0 = %d' % p.predict([0, 0])
