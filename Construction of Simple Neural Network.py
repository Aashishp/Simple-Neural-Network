import numpy as np
import scipy.special
class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes,self.inodes))

        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        
        self.lr = learningrate

        #Using sigmoid function as activation...
        self.activation_func = lambda x:scipy.special.expit(x)
        pass


    def query(self, input_list):
        # conversion of list into 2d array...
        inputs = np.array(input_list, ndmin = 2).T

        #calculate signals emerging from hiddden layer
        hidden_inputs = np.dot(self.who, inputs)

        #calculate signals emerging from final output layer...
        final_outputs = self.activation_func(hidden_inputs)
        
        print(final_outputs)
        return final_outputs

input_nodes = 3
hidden_nodes = 3
output_nodes =2
learning_rate = 0.3

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

n.query([1.0,0.5,-1.5])
