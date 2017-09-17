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


class ImageProcessingLayer(AbstractLayer):
    """
    Layer that is doing actions connected with image processing: convolution, RELU activation function and pooling.
    """

    def __init__(self, input_feature_number, output_feature_number, filter_size, pooling_window_size):
        """
        Initializes filters in this layer.

        :param input_feature_number: number of features of data before this layer
        :param output_feature_number: number of features of data after this layer
        :param filter_size: size of filter
        :param pooling_window_size: size of pooling window
        """
        self.__filters = self.__random_initialize_filters(input_feature_number, output_feature_number, filter_size)
        self.__pooling_window_size = pooling_window_size

    def forward_propagation(self, input_data):
        convolution_output = self.__do_convolution(input_data)
        relu_output = self.__do_relu_activation(convolution_output)
        pooling_output = self.__do_pooling(relu_output)
        return pooling_output

    def backward_propagation(self, input_data):
        return input_data

    def __do_convolution(self, input_data):
        return input_data

    @staticmethod
    def __do_relu_activation(input_data):
        output_data = (input_data > 0) * input_data
        return output_data

    def __do_pooling(self, input_data):
        first_dimension, second_dimension, third_dimension = numpy.shape(input_data)
        reduced_second_dimension = int(numpy.ceil(second_dimension / self.__pooling_window_size))
        reduced_third_dimension = int(numpy.ceil(third_dimension / self.__pooling_window_size))
        output_data_size = tuple([first_dimension, reduced_second_dimension, reduced_third_dimension])
        output_data = numpy.zeros(output_data_size)

        for i in range(0, reduced_second_dimension, 2):
            for j in range(0, reduced_third_dimension, 2):
                max_vector = numpy.amax(input_data[:,
                                        i:i + self.__pooling_window_size,
                                        j:j + self.__pooling_window_size], (1, 2))
                output_data[:, i, j] = max_vector

        return output_data

    @staticmethod
    def __random_initialize_filters(input_feature_number, output_feature_number, filter_size):
        """
        Random initializes filers with numbers in range [-0.5, 0.5]. Size of output filters is as follows:
        output_features x input_features x filter_size x filter_size.

        :param input_feature_number: number of features of data before this layer
        :param output_feature_number: number of features of data after this layer
        :param filter_size: size of filter
        :return: random initialized filters
        """
        random_filters = numpy.random.rand(output_feature_number, input_feature_number, filter_size, filter_size)
        return random_filters


class FullyConnectedLayer(AbstractLayer):
    """
    Fully connected layer where every neuron from one of adjacent layers of data are connected is connected to every
    neuron in second of adjacent layer of data. It does sigmoid activation function.
    """

    def __init__(self, input_neurons_number, output_neurons_number):
        """
        Initializes weights of this layer.

        :param input_neurons_number: number of weights of data before this layer (without biases)
        :param output_neurons_number: number of weights of data after this layer (without biases)
        """
        self.__weights = self.__random_initialize_weights(input_neurons_number, output_neurons_number)

    @staticmethod
    def __random_initialize_weights(input_neurons_number, output_neurons_number):
        """
        Random initializes weights with numbers in range [-0.5, 0.5]. Size of output weights is as follows:
        output_features x input_features.

        :param input_neurons_number: number of weights of data before this layer (without biases)
        :param output_neurons_number: number of weights of data after this layer (without biases)
        :return: random initialized weights
        """
        random_filters = numpy.random.rand(output_neurons_number, input_neurons_number + 1)
        return random_filters

    def forward_propagation(self, input_data):
        return input_data

    def backward_propagation(self, input_data):
        return input_data
