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
        Adds layer to this network. This method returns this object so can be chained with itself.

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

    def learn_network(self, data_to_learn, data_labels):
        """
        Learns this network based on input data.

        :param data_to_learn: data to learn this network on, format of data is matrix of matrices of matrices:\n
            `number of input images x width of single image x height of single image`
        :param data_labels: labels of data, format of this is one-dimensional matrix:\n
            `number of input images x 1`
        """
        data_for_next_layer = self.__normalize_data(data_to_learn)

        for layer in self.__layer_list:
            data_for_next_layer = layer.forward_propagation(data_for_next_layer)
            # print(numpy.shape(data_for_next_layer))

        print(data_for_next_layer)

    def __normalize_data(self, data_to_normalize):
        """
        Normalizes data - transforms them so to range [0, 1].

        :param data_to_normalize: data to process
        :return: normalized data
        """
        max_number = numpy.max(data_to_normalize)
        min_number = numpy.min(data_to_normalize)
        difference = max_number - min_number
        normalized_data = (data_to_normalize - min_number) / difference - 0.5
        return normalized_data


class AbstractNeuralNetworkBuilder(ABC):
    """
    Abstract builder used to build :class:`NeuralNetwork` class
    """

    @abstractmethod
    def set_network_parameters(self, parameters_to_set):
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

    def set_network_parameters(self, parameters_to_set):
        self.__neural_network.set_network_parameters(parameters_to_set)

    def set_layers(self):
        fully_connected_input_neurons = self.__count_fully_connected_input(50)
        self.__neural_network \
            .add_layer(ImageProcessingLayer(1, 50, 5, 2)) \
            .add_layer(ImageProcessingLayer(50, 40, 5, 2)) \
            .add_layer(ImageProcessingLayer(40, 30, 5, 2)) \
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

    def construct(self, network_parameters):
        """
        Construct neural network

        :param network_parameters: parameters of neural network
        :type network_parameters: NeuralNetwork.ParametersContainer
        :return: constructed neural network
        :rtype: NeuralNetwork
        """
        self.__builder.set_network_parameters(network_parameters)
        self.__builder.set_layers()
        result = self.__builder.get_result()
        return result
