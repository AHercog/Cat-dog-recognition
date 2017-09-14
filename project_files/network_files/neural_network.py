from collections import namedtuple

from project_files.network_files.network_layers import *


class NeuralNetwork:
    """
    Class used as neural network. To create instance of this class use :class:`NeuralNetworkDirector`.
    """
    ParametersContainer = namedtuple("ParametersContainer",
                                     ["alpha"])

    def __init__(self):
        self.__layer_list = list()
        self.__network_parameters = None

    def add_layer(self, layer_to_add):
        """
        Adds layer to this network. This method returns this class so can be chained with itself.

        :param layer_to_add: layer to add to network
        :return: self
        """
        self.__layer_list.append(layer_to_add)
        return self

    def set_network_parameters(self, parameters_to_set):
        """
        Sets this layer parameters.

        :param parameters_to_set: parameters to set to this network.
        """
        self.__network_parameters = parameters_to_set


class AbstractNeuralNetworkBuilder(ABC):
    """
    Abstract builder used to build :class:`NeuralNetwork` class
    """

    @abstractmethod
    def set_network_parameters(self):
        raise NotImplementedError

    @abstractmethod
    def set_layers(self):
        raise NotImplementedError

    @abstractmethod
    def get_result(self):
        raise NotImplementedError


class NeuralNetworkBuilder(AbstractNeuralNetworkBuilder):
    def __init__(self):
        self.__neural_network = NeuralNetwork()

    def set_network_parameters(self):
        self.__neural_network.set_network_parameters()

    def set_layers(self):
        fully_connected_input_neurons = self.__count_fully_connected_input(50)
        self.__neural_network \
            .add_layer(ConvolutionalLayer(1, 50, 5)) \
            .add_layer(ReluLayer()) \
            .add_layer(PoolingLayer(2)) \
            .add_layer(ConvolutionalLayer(50, 40, 5)) \
            .add_layer(ReluLayer()) \
            .add_layer(PoolingLayer(2)) \
            .add_layer(ConvolutionalLayer(40, 30, 5)) \
            .add_layer(ReluLayer()) \
            .add_layer(PoolingLayer(2)) \
            .add_layer(FullyConnectedLayer(fully_connected_input_neurons, 25)) \
            .add_layer(FullyConnectedLayer(25, 25)) \
            .add_layer(FullyConnectedLayer(25, 1))

    @staticmethod
    def __count_fully_connected_input(input_map_size_of_network):
        """
        Counts how many neurons will be in first fully connected layer of network.

        :param input_map_size_of_network: size of map on the input of whole network
        :return: number of neurons in first fully connected layer
        """
        map_size_after_first_pooling = numpy.math.floor((input_map_size_of_network - 4) / 2)
        map_size_after_second_pooling = numpy.math.floor((map_size_after_first_pooling - 4) / 2)
        map_size_after_third_pooling = numpy.math.floor((map_size_after_second_pooling - 4) / 2)
        return map_size_after_third_pooling

    def get_result(self):
        return self.__neural_network


class NeuralNetworkDirector:
    def __init__(self, builder):
        """
        Initializes this director with given builder

        :param builder: builder used to build this class
        :type builder: AbstractNeuralNetworkBuilder
        """
        self.__builder = builder

    def construct(self):
        self.__builder.set_network_parameters()
        self.__builder.set_layers()
        result = self.__builder.get_result()
        return result
