import numpy

from project_files.image_parser_files.create_train_data import create_train_data
from project_files.network_files.neural_network import NeuralNetworkDirector, NeuralNetworkBuilder, NeuralNetwork

builder = NeuralNetworkBuilder()
director = NeuralNetworkDirector(builder)
parameters = NeuralNetwork.ParametersContainer(1)
network = director.construct(parameters)

a, b = create_train_data("images\\tiny_set", 50)
network.learn_network(a, b)

# x = numpy.array([[[1, 2, 2], [3, 4, 2]], [[1, 2, 2], [3, 4, 2]]])
# print(numpy.amax(x[:, 0:2, 0:2], (1, 2)))
# z = tuple(int(ti / 2) for ti in numpy.shape(x) )

# a = numpy.zeros(q)
# print(x)
# z = numpy.array([55, 66])
# x[:, 1, 0] = z
# print(x)
# print("\n\n")

# # print(x)
# x = numpy.dot(x[0], numpy.array([[1, 1],[2, 2]]))
# print(x)
#
# x = numpy.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
#
# x = x[0] * numpy.array([[1, 1],[2, 2]])
# print(x)
