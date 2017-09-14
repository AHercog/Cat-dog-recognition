from project_files.network_files.network_layers import *


class NeuralNetwork:
    """
    Class used as neural network. To create instance of this class use NeuralNetworkDirector.
    """
    def __init__(self):
        self.__layer_list = list()
        self.__network_parameters = None

    def add_layer(self, layer_to_add):
        """
        Adds layer to this network. This method returns this class so can be chained with itself.

        :param layer_to_add: layer to add to network
        :type layer_to_add: AbstractLayer
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
