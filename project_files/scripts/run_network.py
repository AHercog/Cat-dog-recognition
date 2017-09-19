import numpy

from project_files.image_parser_files.create_train_data import create_train_data
from project_files.network_files.neural_network import NeuralNetworkDirector, NeuralNetworkBuilder, NeuralNetwork

# building neural network
builder = NeuralNetworkBuilder()
director = NeuralNetworkDirector(builder)
parameters = NeuralNetwork.ParametersContainer(1)
network = director.construct(parameters)

# create data to train and test
train_data_x, train_data_y = create_train_data("images\\train_set", 50)
test_data_x, test_data_y = create_train_data("images\\test_set", 50)

# network learning system isn't working 100% so it is commented
# network.learn_network(a, b)

# propagate test and train data
propagated_train_data = network.propagate_data_through_network(train_data_x)
rounded_propagated_train_data = numpy.round(propagated_train_data)

propagated_test_data = network.propagate_data_through_network(test_data_x)
rounded_propagated_test_data = numpy.round(propagated_test_data)

# count good guessed results of learning
good_train_examples_count = numpy.sum((rounded_propagated_train_data == train_data_y) + 0)
good_test_examples_count = numpy.sum((rounded_propagated_train_data == test_data_y) + 0)

print("{} of {} examples from train set are guessed correctly by network."
      .format(good_train_examples_count, len(train_data_x)))
print("{} of {} examples from test set are guessed correctly by network."
      .format(good_test_examples_count, len(test_data_x)))
