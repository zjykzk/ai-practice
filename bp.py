#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author : zenk
# 2017-03-23 17:12
import math
import random


def sigmoid(x):
    try:
        ret = 1 / (math.expm1(-x) + 2)
        #print('sigmod:%f,ret:%f', x, ret)
        return ret
    except OverflowError:
        print(x)
        return 0 if x < 0 else 1


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
        '''
        print('layer index:%d,node index:%d, %s' %
              (self.layer_index, self.node_index,
               '\n'.join(str(conn) for conn in self.upstream)))
        print('\n'.join(str(conn.upstream_node) for conn in self.upstream))
        '''
        output = reduce(
            lambda ret, conn: ret + conn.upstream_node.output * conn.weight,
            self.upstream, 0)
        #print(output)
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
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str


class ConstNode(Node):
    def __init__(self, layer_index, node_index):
        super(ConstNode, self).__init__(layer_index, node_index)
        self.output = 1

    def append_upstream_connection(self, conn):
        pass

    def calc_output_layer_delta(self, label):
        pass

    def calc_output(self):
        pass

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


class Layer(object):
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


class Connections(object):
    def __init__(self):
        self.connections = []

    def add_connection(self, connection):
        self.connections.append(connection)

    def dump(self):
        for conn in self.connections:
            print(conn)


class Network(object):
    def __init__(self, layers):
        self.layers = [Layer(i, c) for i, c in enumerate(layers)]
        self.connections = Connections()

        for l, nl in zip(self.layers[:-1], self.layers[1:]):
            conns = [Connection(u, d) for u in l.nodes for d in nl.nodes[:-1]]
            for conn in conns:
                self.connections.add_connection(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downsteram_connection(conn)

    def update_weight(self, rate):
        for layer in self.layers[:-1]:
            for n in layer.nodes:
                for conn in n.downstream:
                    conn.update_weight(rate)

    def calc_gradient(self):
        for layer in self.layers[:-1]:
            for n in layer.nodes:
                for conn in n.downstream:
                    conn.calc_gradient()

    def calc_delta(self, labels):
        for n, l in zip(self.layers[-1].nodes, labels):
            n.calc_output_layer_delta(l)

        for layer in self.layers[-2::-1]:
            for n in layer.nodes:
                n.calc_hidden_layer_delta()

    def predict(self, sample):
        self.layers[0].set_output(sample)
        for layer in self.layers[1:]:
            layer.calc_output()

        '''
        for l in self.layers:
            l.dump()
        '''

        return map(lambda n: n.output, self.layers[-1].nodes[:-1])

    def get_gradient(self, labels, sample):
        self.predict(sample)
        self.calc_delta(labels)
        self.calc_gradient()

    def train(self, labels, samples, rate, iteration):
        for _ in range(iteration):
            for label, sample in zip(labels, samples):
                self.train_one_sample(label, sample, rate)

    def train_one_sample(self, labels, sample, rate):
        self.predict(sample)
        self.calc_delta(labels)
        #self.dump()
        self.update_weight(rate)
        #self.dump()

    def dump(self):
        print('---------------------------------dump network')
        for layer in self.layers:
            layer.dump()
        print('+++++++++++++++++++++++++++++++++dump network')


def gradient_check(network, sample_feature, sample_label):
    network_error = lambda vec1, vec2: \
        .5 * sum(map(lambda v: (v[0] - v[1]) * (v[0] - v[1]), zip(vec1, vec2)))

    network.get_gradient(sample_feature, sample_label)

    for conn in network.connections.connections:
        epsilon = 0.0001
        conn.weight += epsilon
        error1 = network_error(network.predict(sample_feature), sample_label)

        conn.weight -= 2 * epsilon
        error2 = network_error(network.predict(sample_feature), sample_label)

        expected_gradient = (error2 - error1) / (2 * epsilon)

        print 'expected gradient:\t%f\nactual gradient: \t%f' % (
            expected_gradient, conn.get_gradient())


def test():
    input_vecs = [[10], [20], [31], [44], [49]]
    labels = [[1000], [2000], [3600], [4000], [5000]]
    #input_vecs = [[10], [20]]
    #labels = [[1000], [2000]]
    network = Network([1, 1, 1])
    network.train(labels, input_vecs, 0.3, 100)
    for f, l in zip(input_vecs, labels):
        gradient_check(network, f, l)

    print(network.predict([90]))
    print(network.predict([10]))


if __name__ == '__main__':
    test()
