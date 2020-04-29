import torch


class Module(object):
    def forward(self, *inputs):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return []


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
        return self.weights @ inputs + self.bias

    def backward(self, gradwrtoutput: torch.Tensor) -> torch.Tensor:
        dx = self.weights.t() @ gradwrtoutput
        self.dbias.add_(gradwrtoutput)
        self.dweights.add_(gradwrtoutput.view((-1, 1)).mm(self.input.view(1, -1)))
        return dx

    def param(self) -> list:
        return [(self.weights, self.dweights), (self.bias, self.dbias)]


class ReLU(Module):

    def __init__(self):
        self.input: torch.Tensor = torch.empty(0)
        self.drelu: torch.Tensor = torch.empty(0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        self.input = inputs
        inputs = inputs.view((-1, 1))
        zeros = torch.zeros_like(inputs)
        conc = torch.cat((inputs, zeros), 1)
        act = conc.max(1)
        self.drelu = (1 - act[1]).float()
        return act[0]

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

    def param(self):
        return []

class LossMSE(Module):

    def __init__(self):
        self.prediction: torch.Tensor = torch.empty(0)
        self.target: torch.Tensor = torch.empty(0)

    def forward(self, *inputs):
        self.prediction = inputs[0]
        self.target = inputs[1]
        return torch.sum(torch.pow(self.prediction-self.target, 2))

    def backward(self):
        dloss = 2 * (self.prediction - self.target)
        return dloss

    def param(self):
        return []

