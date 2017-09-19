from collections import namedtuple

from project_files.network_files.network_layers import *


class NeuralNetwork:
    """
    Class used as neural network. To create instance of this class use :class:`NeuralNetworkDirector`.
    """

    # mini class used to hold parameters used to create neural network
    ParametersContainer = namedtuple("ParametersContainer", ["alpha"])

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

    def learn_network(self, input_data, data_labels):
        """
        Learns this network based on input data.

        :param input_data: data to learn this network on, format of data is matrix of matrices of matrices:\n
            `number of input images x width of single image x height of single image`
        :param data_labels: labels of data, format of this is one-dimensional matrix:\n
            `number of input images x 1`
        """
        normalized_data = self.__normalize_data(input_data)
        data_from_forward_propagation = self.__do_forward_propagation(normalized_data)
        last_layer_output = data_from_forward_propagation[-1].data_after_activation
        last_layer_deltas = last_layer_output - data_labels
        delta_list = self.__do_backward_propagation(last_layer_deltas, data_from_forward_propagation)

        # for i in data_from_forward_propagation:
        #     print(numpy.shape(i.data_before_activation))
        #     print(numpy.shape(i.data_after_activation))
        #
        # for i in delta_list:
        #     print(numpy.shape(i))

    def propagate_data_through_network(self, input_data):
        data_for_next_layer = self.__normalize_data(input_data)

        for layer in self.__layer_list:
            propagated_data = layer.forward_propagation(data_for_next_layer)
            data_for_next_layer = propagated_data.data_after_activation

        return data_for_next_layer

    def __do_forward_propagation(self, input_data):
        """
        Does forward propagation pass for whole network.

        :param input_data: data to make forward pass on
        :return: list of results of every layer
        :rtype: list of AbstractLayer.ForwardPropagationData
        """
        result_list = list()
        data_for_next_layer = input_data

        for layer in self.__layer_list:
            propagated_data = layer.forward_propagation(data_for_next_layer)
            data_for_next_layer = propagated_data.data_after_activation
            result_list.append(propagated_data)
            # print(numpy.shape(data_for_next_layer))

        return result_list

    def __do_backward_propagation(self, input_data, forward_data_list):
        """
        Does backward propagation pass for whole network.

        :param input_data: data to make backward pass on
        :param forward_data_list: data of every layer from forward propagation
        :return: list deltas for every layer
        """
        result_list = list()
        data_for_next_layer = input_data

        for layer_index in range(len(self.__layer_list) - 1, 0, -1):
            layer = self.__layer_list[layer_index]
            data_for_next_layer = layer.backward_propagation(data_for_next_layer,
                                                             forward_data_list[layer_index - 1].data_before_activation)
            result_list.append(data_for_next_layer)

        result_list.reverse()
        return result_list

    @staticmethod
    def __normalize_data(data_to_normalize):
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
        map_size_after_first_pooling = numpy.math.ceil((input_map_size_of_network - 4) / 2)
        map_size_after_second_pooling = numpy.math.ceil((map_size_after_first_pooling - 4) / 2)
        map_size_after_third_pooling = numpy.math.ceil((map_size_after_second_pooling - 4) / 2)
        fully_connected_size = 30 * map_size_after_third_pooling * map_size_after_third_pooling
        return fully_connected_size

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
        Constructs neural network

        :param network_parameters: parameters of neural network
        :type network_parameters: NeuralNetwork.ParametersContainer
        :return: constructed neural network
        :rtype: NeuralNetwork
        """
        self.__builder.set_network_parameters(network_parameters)
        self.__builder.set_layers()
        return self.__builder.get_result()
