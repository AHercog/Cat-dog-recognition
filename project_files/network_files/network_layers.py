"""
Module containing all types of layers used in neural network.
"""
from abc import ABC, abstractmethod
from collections import namedtuple
from time import sleep

import numpy

from project_files.network_files.activation_functions import ReluFunction, SigmoidFunction


class AbstractLayer(ABC):
    """
    Abstract base class for all types of layers in neural network.
    """

    # sub-class used to pack data returned by forward propagation
    ForwardPropagationData = namedtuple("ForwardPropagationData",
                                        ["data_before_activation", "data_after_activation"])

    @abstractmethod
    def forward_propagation(self, input_data):
        """
        Does forward pass through this layer and returns tuple of outputs: one before and one after activation function.

        :param input_data: data on which make forward pass
        :return: data before activation function, data after activation function
        :rtype: AbstractLayer.ForwardPropagationData
        """
        raise NotImplementedError

    @abstractmethod
    def backward_propagation(self, input_data, forward_data):
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
        self.__input_map_size = None

    def forward_propagation(self, input_data):
        convolution_output = self.__do_forward_convolution(input_data)
        pooling_output = self.__do_forward_pooling(convolution_output)
        relu_output = self.__do_forward_relu_activation(pooling_output)
        output_data = AbstractLayer.ForwardPropagationData(pooling_output, relu_output)
        return output_data

    def backward_propagation(self, input_data, forward_data):
        if not self.__is_data_image_like(input_data):
            input_data = self.__make_data_image_like(input_data)

        convolution_output = self.__do_backward_convolution(input_data)
        pooling_output = self.__do_backward_pooling(convolution_output)
        relu_output = self.__do_backward_relu_activation(pooling_output, forward_data)
        return relu_output

    def __do_forward_convolution(self, input_data):
        number_of_examples, input_map_count, input_map_size, input_map_size = numpy.shape(input_data)
        output_map_count, input_map_count, filter_size, filter_size = numpy.shape(self.__filters)
        output_map_size = input_map_size - filter_size + 1
        output_data = numpy.zeros((number_of_examples, output_map_count, output_map_size, output_map_size))
        self.__input_map_size = input_map_size

        for actual_output_map in range(output_map_count):
            for actual_output_y in range(output_map_size):
                for actual_output_x in range(output_map_size):
                    multiplied_matrices = input_data[:, :,
                                          actual_output_y: actual_output_y + filter_size,
                                          actual_output_x: actual_output_x + filter_size] \
                                          * self.__filters[actual_output_map, :, :, :]
                    summed_result = numpy.sum(multiplied_matrices, (1, 2, 3))
                    divide_number = input_map_count * filter_size * filter_size
                    output_data[:, actual_output_map, actual_output_y, actual_output_x] = summed_result / divide_number

        return output_data

    def __do_backward_convolution(self, input_data):
        number_of_examples, input_map_count, input_map_size, input_map_size = numpy.shape(input_data)
        input_map_count, output_map_count, filter_size, filter_size = numpy.shape(self.__filters)
        output_map_size = self.__input_map_size
        output_data = numpy.zeros((number_of_examples, output_map_count, output_map_size, output_map_size))

        for actual_output_map in range(output_map_count):
            for actual_filter_y in range(output_map_size):
                for actual_filter_x in range(output_map_size):
                    sum_of_elements = 0

                    for actual_input_map in range(input_map_count):
                        for i in range(filter_size):
                            for j in range(filter_size):
                                if 0 < actual_filter_y - i < input_map_size \
                                        and 0 < actual_filter_x - j < input_map_size:
                                    sum_of_elements += input_data[
                                                       :, actual_input_map, actual_filter_y - i, actual_filter_x - j] \
                                                       * self.__filters[
                                                           actual_input_map, actual_output_map, i, j]

                    sum_of_elements /= (input_map_count * filter_size * filter_size)
                    output_data[:, actual_output_map, actual_filter_y, actual_filter_x] = sum_of_elements

        return output_data

    def __do_forward_pooling(self, input_data):
        """
        Does pooling by taking maximum value from every window of pre-defined size.

        :param input_data: data to do pooling on
        :return: data shrunk by pooling
        """
        first_dimension, second_dimension, third_dimension, fourth_dimension = numpy.shape(input_data)
        reduced_third_dimension = int(numpy.ceil(third_dimension / self.__pooling_window_size))
        reduced_fourth_dimension = int(numpy.ceil(fourth_dimension / self.__pooling_window_size))
        output_data_size = tuple([first_dimension, second_dimension, reduced_third_dimension, reduced_fourth_dimension])

        output_data = numpy.zeros(output_data_size)
        chosen_units_map = numpy.zeros(numpy.shape(input_data))

        for height_index in range(0, reduced_third_dimension, 2):
            for width_index in range(0, reduced_fourth_dimension, 2):
                max_vector = numpy.amax(input_data[:, :,
                                        height_index:height_index + self.__pooling_window_size,
                                        width_index:width_index + self.__pooling_window_size], (2, 3))
                max_indices_vector = numpy.argmax(input_data[:, :,
                                                  height_index:height_index + self.__pooling_window_size,
                                                  width_index:width_index + self.__pooling_window_size], 3)

                output_data[:, :, height_index, width_index] = max_vector

        return output_data

    def __do_backward_pooling(self, input_data):
        return input_data

    @staticmethod
    def __do_forward_relu_activation(input_data):
        output_data = ReluFunction.calculate(input_data)
        return output_data

    @staticmethod
    def __do_backward_relu_activation(input_data, forward_data):
        relu_gradient = ReluFunction.calculate_gradient(forward_data)
        output_data = input_data * relu_gradient
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
        random_filters = numpy.random.rand(output_feature_number, input_feature_number, filter_size, filter_size) - 0.5
        return random_filters

    @staticmethod
    def __is_data_image_like(input_data):
        """
        Checks input data is image-like (ready for process for convolutional layer or not).

        :param input_data: data to check if is image-like
        :return: is data image-like
        :rtype: bool
        """
        input_data_size_tuple = numpy.shape(input_data)
        input_data_size = sum(1 for _ in input_data_size_tuple)

        if input_data_size == 4:
            return True
        else:
            return False

    def __make_data_image_like(self, input_data):
        """
        Flattens data and adds bias unit so they can be processed by fully connected layer.

        :param input_data: data to flatten
        :return: flattened data
        """
        number_of_images, neuron_count = numpy.shape(input_data)
        output_map_count, input_map_count, filter_size, filter_size = numpy.shape(self.__filters)
        image_like_data = numpy.reshape(input_data, (number_of_images, output_map_count, 3, 3))
        return image_like_data


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

    def forward_propagation(self, input_data):
        if not self.__is_data_flattened(input_data):
            input_data = self.__flatten_data(input_data)

        fully_connected_output = self.__do_forward_fully_connected(input_data)
        sigmoid_output = self.__do_forward_sigmoid_activation(fully_connected_output)
        output_data = AbstractLayer.ForwardPropagationData(fully_connected_output, sigmoid_output)
        return output_data

    def backward_propagation(self, input_data, forward_data):
        fully_connected_output = self.__do_backward_fully_connected(input_data)
        sigmoid_output = self.__do_backward_sigmoid_activation(fully_connected_output, forward_data)
        return sigmoid_output

    def __do_forward_fully_connected(self, input_data):
        output_data = numpy.dot(input_data, numpy.transpose(self.__weights))
        return output_data

    def __do_backward_fully_connected(self, input_data):
        output_data = numpy.dot(input_data, self.__weights)
        return output_data

    @staticmethod
    def __is_data_flattened(input_data):
        """
        Checks input data is flattened (ready for process for fully connected layer or not).

        :param input_data: data to check if is flattened
        :return: is data flattened
        :rtype: bool
        """
        input_data_size_tuple = numpy.shape(input_data)
        input_data_size = sum(1 for _ in input_data_size_tuple)

        if input_data_size == 2:
            return True
        else:
            return False

    @staticmethod
    def __flatten_data(input_data):
        """
        Flattens data and adds bias unit so they can be processed by fully connected layer.

        :param input_data: data to flatten
        :return: flattened data
        """
        number_of_images, map_count, filter_size, filter_size = numpy.shape(input_data)
        flattened_size = map_count * filter_size * filter_size
        flattened_data = numpy.reshape(input_data, (number_of_images, flattened_size))
        return flattened_data

    @staticmethod
    def __do_forward_sigmoid_activation(input_data):
        output_data = SigmoidFunction.calculate(input_data)
        return output_data

    def __do_backward_sigmoid_activation(self, input_data, forward_data):
        if not self.__is_data_flattened(forward_data):
            flattened_forward_data = self.__flatten_data(forward_data)
        else:
            flattened_forward_data = forward_data

        sigmoid_gradient = SigmoidFunction.calculate_gradient(flattened_forward_data)
        output_data = input_data * sigmoid_gradient
        return output_data

    @staticmethod
    def __random_initialize_weights(input_neurons_number, output_neurons_number):
        """
        Random initializes weights with numbers in range [-0.5, 0.5]. Size of output weights is as follows:
        output_features x input_features.

        :param input_neurons_number: number of weights of data before this layer (without biases)
        :param output_neurons_number: number of weights of data after this layer (without biases)
        :return: random initialized weights
        """
        random_weights = numpy.random.rand(output_neurons_number, input_neurons_number) - 0.5
        return random_weights
