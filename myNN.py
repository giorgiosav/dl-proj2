import torch
from functools import reduce

class Module(object):
    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []

    def zero_grad(self):
        pass

    def __call__(self, *args):
        return self.forward(*args)


class Linear(Module):

    def __init__(self, nb_hidden1: int, nb_hidden2: int):
        self.weights: torch.Tensor = torch.empty(nb_hidden2, nb_hidden1).normal_()
        self.bias: torch.Tensor = torch.empty(nb_hidden2).normal_()
        self.dweights: torch.Tensor = torch.empty(nb_hidden2, nb_hidden1)
        self.dbias: torch.Tensor = torch.empty(nb_hidden2)
        self.input: torch.Tensor = torch.empty(nb_hidden1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        :param inputs:
        :return:
        """
        self.input = inputs

        # optimized version of bias + input @ weights.t()
        return torch.addmm(self.bias, self.input, self.weights.t())

    def backward(self, gradwrtoutput: torch.Tensor) -> torch.Tensor:
        dx = gradwrtoutput @ self.weights
        self.dbias.add_(gradwrtoutput.sum(0))
        self.dweights.add_(gradwrtoutput.t().mm(self.input))
        return dx

    def param(self) -> list:
        return [(self.weights, self.dweights), (self.bias, self.dbias)]

    def zero_grad(self):
        self.dweights = torch.zeros(self.dweights.shape)
        self.dbias = torch.zeros(self.dbias.shape)


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
        
        self.n_elements = reduce(lambda a,b: a*b, self.prediction.shape)

        return torch.sum(torch.pow(self.prediction-self.target, 2)) / self.n_elements

    def backward(self):
        dloss = 2 * (self.prediction - self.target) / self.n_elements
        return dloss

    def param(self):
        return []

