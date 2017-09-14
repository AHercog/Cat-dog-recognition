"""
Module containing all types of layers used in neural network.
"""
from abc import ABC, abstractmethod

import numpy


class AbstractLayer(ABC):
    """
    Abstract base class for all types of layers in neural network.
    """

    @abstractmethod
    def forward_propagation(self, input_data):
        """
        Does forward pass through this layer and returns output.

        :param input_data: data on which make forward pass
        :return: output of this layer
        """
        raise NotImplementedError

    @abstractmethod
    def backward_propagation(self, input_data):
        """
        Does backward pass through this layer and return output.

        :param input_data: data on which make backward pass
        :return: output of this layer
        """
        raise NotImplementedError


class ConvolutionalLayer(AbstractLayer):
    """
    Layer that is doing convolution on given data.
    """

    def __init__(self, input_feature_number, output_feature_number, filter_size):
        self.__filters = self.__random_initialize_filters(input_feature_number, output_feature_number, filter_size)

    @staticmethod
    def __random_initialize_filters(input_feature_number, output_feature_number, filter_size):
        """
        Random initializes filers with numbers in range [-0.5, 0.5]. Size of output filters is as follows:
        output_features x input_features x filter_size x filter_size

        :param input_feature_number: number of features of data before this layer
        :param output_feature_number: number of features of data after this layer
        :param filter_size: size of filter
        :return: random initialized filters
        """
        random_filters = numpy.random.rand(output_feature_number, input_feature_number, filter_size, filter_size)
        return random_filters

    def forward_propagation(self, input_data):
        return input_data

    def backward_propagation(self, input_data):
        return input_data


class ReluLayer(AbstractLayer):
    """
    Layer that is applying relu function to given data.
    """

    def forward_propagation(self, input_data):
        return input_data

    def backward_propagation(self, input_data):
        return input_data


class PoolingLayer(AbstractLayer):
    """
    Layer that does pooling on given data.
    """

    def __init__(self, pooling_window_size):
        self.__pooling_window_size = pooling_window_size

    def forward_propagation(self, input_data):
        return input_data

    def backward_propagation(self, input_data):
        return input_data


class FullyConnectedLayer(AbstractLayer):
    """
    Fully connected layer where every neuron from one of adjacent layers of data are connected is connected to every
    neuron in second of adjacent layer of data.
    """

    def __init__(self, input_neurons_number, output_neurons_number):
        self.__weights = self.__random_initialize_weights(input_neurons_number, output_neurons_number)

    @staticmethod
    def __random_initialize_weights(input_neurons_number, output_neurons_number):
        """
        Random initializes weights with numbers in range [-0.5, 0.5]. Size of output weights is as follows:
        output_features x input_features

        :param input_neurons_number: number of weights of data before this layer (without biases)
        :param output_neurons_number: number of weights of data after this layer (without biases)
        :return: random initialized weights
        """
        random_filters = numpy.random.rand(output_neurons_number, input_neurons_number)
        return random_filters

    def forward_propagation(self, input_data):
        return input_data

    def backward_propagation(self, input_data):
        return input_data


class SigmoidLayer(AbstractLayer):
    """
    Layer that is applying sigmoid function to given data.
    """

    def forward_propagation(self, input_data):
        return input_data

    def backward_propagation(self, input_data):
        return input_data
