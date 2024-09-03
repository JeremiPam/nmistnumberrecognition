import numpy
import matplotlib.pyplot
import scipy.special

data_file = open("mnist dataset/mnist_train_100.csv", 'r')#open function only works with little files
data_list = data_file.readlines()
data_file.close()






class neuralNetwork:
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.iNodes = inputNodes
        self.hNodes = hiddenNodes
        self.oNodes = outputNodes
        self.lr = learningRate
        self.wih = numpy.random.rand(self.hNodes, self.iNodes) - 0.5
        self.who = numpy.random.rand(self.oNodes, self.hNodes) - 0.5
        self.activation_function = lambda x: scipy.special.expit(x)

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.lr * numpy.dot(output_errors * final_outputs * (1.0 - final_outputs), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot(hidden_errors * hidden_outputs * (1.0-hidden_outputs),numpy.transpose(inputs))

input_nodes = 784
hidden_nodes = 100
output_nodes = 10

n = neuralNetwork(input_nodes, hidden_nodes,output_nodes, 0.3)
all_values = data_list[7].split(',')
inputs = (numpy.asarray(all_values[1:],dtype=numpy.float64) / 255.0 * 0.99) +0.01
image_array = numpy.asarray(all_values[1:],dtype=numpy.float64).reshape((28,28))
targets = numpy.zeros(output_nodes) + 0.01

targets[int(all_values[0])] = 0.99
n.train(inputs, targets)

print(n.query((numpy.asarray(all_values[1:],dtype=numpy.float64)/ 255.0 * 0.99)+0.01))

matplotlib.pyplot.imshow(image_array, cmap='Greys',interpolation='None')
matplotlib.pyplot.show()