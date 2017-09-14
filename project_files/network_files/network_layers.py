"""
Module containing all types of layers used in neural network.
"""
from abc import ABC, abstractmethod


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
    def forward_propagation(self, input_data):
        return input_data

    def backward_propagation(self, input_data):
        return input_data


class FullyConnectedLayer(AbstractLayer):
    """
    Fully connected layer where every neuron from one of adjacent layers of data are connected is connected to every
    neuron in second of adjacent layer of data.
    """
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
