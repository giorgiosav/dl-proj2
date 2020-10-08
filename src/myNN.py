# -*- coding: utf-8 -*-
"""Package to define different Module/combination of them (emulate torch.nn package)"""

import torch
from functools import reduce
import math


class Module(object):
    """
    Superclass for all implemented modules.
    """

    def forward(self, *inputs):
        """Forward pass"""
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        """Backward pass"""
        raise NotImplementedError

    def param(self):
        """
        Modules with no params return an empty list, the others
        overwrite this function.
        """
        return []

    def zero_grad(self):
        """
        Modules that don't define this function do nothing.
        """
        pass

    def __call__(self, *args):
        """
        Enables pytorch-like syntax of calling a module directly.
        """
        return self.forward(*args)


class Linear(Module):
    """
    Implements linear fully-connected layer
    """

    def __init__(self, nb_hidden1: int, nb_hidden2: int):
        """
        Performs initialization of weights and biases like in pytoch,
        according to a uniform distribution. This helps with convergence
        during training.
        :param nb_hidden1: number of input hidden units
        :param nb_hidden2: number of output hidden units
        """
        # Init 1: Pytorch Init
        std = 1.0 / math.sqrt(nb_hidden1)
        self.weights: torch.Tensor = torch.empty(nb_hidden2, nb_hidden1).uniform_(-std, std)
        self.bias: torch.Tensor = torch.empty(nb_hidden2).uniform_(-std, std)
        self.dweights: torch.Tensor = torch.empty(nb_hidden2, nb_hidden1)
        self.dbias: torch.Tensor = torch.empty(nb_hidden2)
        self.input: torch.Tensor = torch.empty(nb_hidden1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Applies weights and biases to input data and returns result.
        :param inputs: input data batch
        :return: output data batch
        """
        self.input = inputs

        # optimized version of bias + input @ weights.t()
        return torch.addmm(self.bias, self.input, self.weights.t())

    def backward(self, gradwrtoutput: torch.Tensor) -> torch.Tensor:
        """
        Calculates gradients of loss wrt weights and biases and accumulates them.
        :param gradwrtoutput: batch of loss gradients wrt output
        :return: loss gradients wrt input
        """
        dx = gradwrtoutput @ self.weights
        self.dbias.add_(gradwrtoutput.sum(0))  # explicit sum over batches
        self.dweights.add_(gradwrtoutput.t().mm(self.input))  # implicit sum over batches
        return dx

    def param(self) -> list:
        """
        Get layer parameters
        :return: a list of 2 tuples. The first tuple contains weights and
                the gradients of the loss wrt to the weights. The second tuple contains
                the same but for biases.
        """
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
    """
    Implements ReLU layer
    """

    def __init__(self):
        """Initialize ReLU layer"""

        self.input: torch.Tensor = torch.empty(0)
        self.drelu: torch.Tensor = torch.empty(0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Apply forward, saves input and calculates derivative
        :param inputs: input data batch
        :return: output data batch (with ReLU applied)
        """
        self.input = inputs

        # Apply relu and compute its derivative
        relu = torch.relu(self.input)
        self.drelu = relu.clone()
        self.drelu[self.drelu != 0] = 1  # derivative

        return relu

    def backward(self, gradwrtoutput: torch.Tensor) -> torch.Tensor:
        """
        Apply ReLU activation backward
        :param gradwrtoutput: batch of loss gradients wrt output
        :return: loss gradients wrt input
        """
        return self.drelu * gradwrtoutput

    def param(self) -> list:
        return []


class Tanh(Module):
    """
    Implements Tanh layer
    """

    def __init__(self):
        """Init Tanh"""
        self.input: torch.Tensor = torch.empty(0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Do forward pass, save input
        :param inputs: input data batch
        :return: output data batch (with Tanh applied)
        """
        self.input = inputs
        return torch.tanh(inputs)

    def backward(self, gradwrtoutput: torch.Tensor) -> torch.Tensor:
        """
        Do backward of Tanh activation
        :param gradwrtoutput: batch of loss gradients wrt output
        :return: loss gradients wrt input
        """
        dtanh = 1 - torch.pow(torch.tanh(self.input), 2)
        return dtanh * gradwrtoutput

    def param(self) -> list:
        return []


class Sequential(Module):
    """
    Implements sequential module with multiple layers
    """

    def __init__(self, *layers: Module):
        """
        Init Sequential, save list of module (and reverse for backward)
        :param layers: layers in the module
        """
        self.layers: list[Module] = list(layers)
        self.backlayers: list[Module] = list(layers)
        self.backlayers.reverse()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Do forward pass traversing all layers
        :param inputs: input data batch
        :return: output data batch (with all layers applied)
        """
        x = inputs
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self, gradwrtoutput: torch.Tensor) -> torch.Tensor:
        """
        Calls backward() for every layer in the module
        :param gradwrtoutput: (torch.Tensor) batch of loss gradients wrt output
        :return: (torch.Tensor) loss gradients wrt module input 
        """
        x = gradwrtoutput
        for l in self.backlayers:
            x = l.backward(x)

        return x

    def zero_grad(self):
        """
        Sets all gradients in the module to zero, by calling zero_grad on
        every layer.
        """
        for l in self.layers:
            l.zero_grad()

    def param(self) -> list:
        """
        :return: list of params for every layer
        """
        return [l.param() for l in self.layers]


class LossMSE(Module):
    """
    Implements MSE loss module
    """

    def __init__(self):
        """Init LossMSE params"""
        self.prediction: torch.Tensor = torch.empty(0)
        self.target: torch.Tensor = torch.empty(0)
        self.n_elements = 0

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> float:
        """
        MSE loss is averaged across all batches, as is the default behavior
        in Pytorch.
        :param prediction: input tensor to compare
        :param target: input target tensor
        :return: MSE loss between prediction and target
        """
        self.prediction = prediction
        self.target = target

        # Check shapes
        if self.prediction.shape != self.target.shape:
            raise ValueError("Shape mismatch, prediction: {}, target: {}".format(self.prediction.shape, self.target.shape))

        # Save number of elements in the batch for backward
        self.n_elements = reduce(lambda a, b: a * b, self.prediction.shape)

        # Mean as in pytorch
        return torch.mean((self.prediction - self.target) ** 2)

    def backward(self) -> torch.Tensor:
        """
        The gradient is divided by the number of elements (across all batches),
        as is the default in Pytorch.
        :return: gradient of the MSE loss wrt the input prediction.
        """
        dloss = 2 * (self.prediction - self.target) / self.n_elements
        return dloss

    def param(self) -> list:
        return []
