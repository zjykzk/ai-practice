#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author : zenk
# 2017-03-23 17:12
import math
import random


def sigmoid(x):
    -1 / math.expm1(-x)


class Node(object):
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0
        self.delta = 0

    def set_output(self, output):
        self.output = output

    def append_downsteram_connection(self, conn):
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        self.upstream.append(conn)

    def calc_output(self):
        output = reduce(
            lambda ret, conn: ret + conn.upstream_node.output * conn.weight,
            self.upstream, 0)
        self.output = sigmoid(output)

    def calc_hidden_layer_delta(self):
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def calc_output_layer_delta(self, label):
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index,
                                                    self.node_index,
                                                    self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn),
                                self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn),
                              self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\t' + upstream_str


class ConstNode(object):
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1

    def append_downsteram_connection(self, conn):
        self.downstream.append(conn)

    def calc_hidden_layer_delta(self):
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def __str__(self):
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn),
                                self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str


def Layer(object):
    def __init__(self, layer_index, node_count):
        self.layer_index = layer_index
        self.nodes = [Node(layer_index, i) for i in range(node_count)]
        self.nodes.append(ConstNode(layer_index, node_count))

    def set_output(self, data):
        for d, n in zip(data, self.nodes):
            n.set_output(d)

    def calc_output(self):
        for n in self.nodes[:-1]:
            n.calc_output()

    def dump(self):
        for n in self.nodes:
            print(n)


class Connection(object):
    def __init__(self, upstream_node, downstream_node):
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-.1, .1)
        self.gradient = 0.0

    def calc_gradient(self):
        self.gradient = self.downstream_node.delta * self.upstream_node.output

    def get_gradient(self):
        return self.gradient

    def update_weight(self, rate):
        self.calc_gradient()
        self.weight += rate * self.gradient

    def __str__(self):
        return '(%u-%u) -> (%u-%u) - %f' % (self.upstream_node.layer_index,
                                            self.upstream_node.node_index,
                                            self.downstream_node.layer_index,
                                            self.downstream_node.node_index,
                                            self.weight)
