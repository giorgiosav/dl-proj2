import torch
from functools import reduce
import math


class Module(object):
    '''
    Superclass for all implemented modules.
    '''
    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        '''
        Modules with no params return an empty list, the others
        overwrite this function.
        '''
        return []

    def zero_grad(self):
        '''
        Modules that don't define this function do nothing.
        '''
        pass

    def __call__(self, *args):
        '''
        Enables pytorch-like syntax of calling a module directly.
        '''
        return self.forward(*args)


class Linear(Module):
    '''
    Implements linear fully-connected layer
    '''
    def __init__(self, nb_hidden1: int, nb_hidden2: int):
        '''
        :param nb_hidden1: (int) number of input hidden units
        :param nb_hidden2: (int) number of output hidden units

        Performs initialization of weights and biases like in pytoch, 
        according to a uniform distribution. This helps with convergence
        during training.
        '''
        # Init 1: Pytorch Init
        std = 1. / math.sqrt(nb_hidden1)
        self.weights: torch.Tensor = torch.empty(nb_hidden2, nb_hidden1)\
                                          .uniform_(-std, std)
        self.bias: torch.Tensor = torch.empty(nb_hidden2).uniform_(-std, std)
        self.dweights: torch.Tensor = torch.empty(nb_hidden2, nb_hidden1)
        self.dbias: torch.Tensor = torch.empty(nb_hidden2)
        self.input: torch.Tensor = torch.empty(nb_hidden1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs: (torch.Tensor) input data batch
        :return: (torch.Tensor) output data batch

        Applies weights and biases to input data and returns result.
        """
        self.input = inputs

        # optimized version of bias + input @ weights.t()
        return torch.addmm(self.bias, self.input, self.weights.t())

    def backward(self, gradwrtoutput: torch.Tensor) -> torch.Tensor:
        '''
        :param gradwrtoutput: (torch.Tensor) batch of gradients wrt output
        :return: (torch.Tensor)
        '''
        dx = gradwrtoutput @ self.weights
        self.dbias.add_(gradwrtoutput.sum(0))
        self.dweights.add_(gradwrtoutput.t().mm(self.input))
        return dx

    def param(self) -> list:
        return [(self.weights, self.dweights), (self.bias, self.dbias)]

    def zero_grad(self):
        """
        Zero-out the gradients. This is strange syntax, but you cannot
        reassign self.dweights and self.dbias to a new all-zero tensor.
        If you do, then the optimizer will lose the reference to dweights
        and dbias.
        """
        self.dweights[True] = 0
        self.dbias[True] = 0


class ReLU(Module):

    def __init__(self):
        self.input: torch.Tensor = torch.empty(0)
        self.drelu: torch.Tensor = torch.empty(0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.input = inputs

        relu = torch.relu(self.input)
        self.drelu = relu.clone()
        self.drelu[self.drelu != 0] = 1

        return relu

    def backward(self, gradwrtoutput: torch.Tensor) -> torch.Tensor:
        return self.drelu * gradwrtoutput

    def param(self):
        return []


class Tanh(Module):

    def __init__(self):
        self.input: torch.Tensor = torch.empty(0)

    def forward(self, inputs):
        self.input = inputs
        return torch.tanh(inputs)

    def backward(self, gradwrtoutput):
        dtanh = 1 - torch.pow(torch.tanh(self.input), 2)
        return dtanh * gradwrtoutput

    def param(self):
        return []


class Sequential(Module):

    def __init__(self, *layers):
        self.layers: list[Module] = list(layers)
        self.backlayers: list[Module] = list(layers)
        self.backlayers.reverse()

    def forward(self, inputs):
        x = inputs
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self, gradwrtoutput):
        x = gradwrtoutput
        for l in self.backlayers:
            x = l.backward(x)

        return x

    def zero_grad(self):
        for l in self.layers:
            l.zero_grad()

    def param(self):
        return [l.param() for l in self.layers]


class LossMSE(Module):

    def __init__(self):
        self.prediction: torch.Tensor = torch.empty(0)
        self.target: torch.Tensor = torch.empty(0)
        self.n_elements = 0

    def forward(self, prediction, target):
        self.prediction = prediction
        self.target = target

        if self.prediction.shape != self.target.shape:
            raise ValueError(
                "Shape mismatch, prediction: {}, target: {}".format(
                    self.prediction.shape, self.target.shape
                )
            )

        self.n_elements = reduce(lambda a, b: a * b, self.prediction.shape)

        return torch.mean((self.prediction - self.target)**2)

    def backward(self):
        dloss = 2 * (self.prediction - self.target) / self.n_elements
        return dloss

    def param(self):
        return []
