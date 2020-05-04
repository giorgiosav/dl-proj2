# -*- coding: utf-8 -*-
"""Package to define optimizers"""


class SGD:
    def __init__(self, params: list, lr: float, momentum: float = 0):
        """
        Save params to optimize them in step
        Python passes params by reference - just make sure not to reassign
        the params in your module.
        Note: momentum implementation is same as pytorch
        :param params: params of the module
        :param lr: learning rate (eta)
        :param momentum: momentum factor
        """
        self.params = params
        self.lr = lr
        self.momentum = momentum
        # First velocity factor is the original derivative (0)
        if self.momentum != 0:
            self.vt_prev = []
            for tuple_par in params:
                for _, dx in tuple_par:
                    self.vt_prev.append(dx)

    def step(self):
        """
        Performs an SGD step, by updating the weights and biases
        in the module according to the specified learning rate and
        momentum.
        """
        # Cycle over each param and derivative, updating
        for (tuple_index, tuple_par) in enumerate(self.params):
            for param_index, (x, dx) in enumerate(tuple_par):
                i = tuple_index + param_index
                if self.momentum != 0:
                    # Velocity computation and momentum update
                    vt = self.vt_prev[i] * self.momentum + dx
                    self.vt_prev[i] = vt
                    x -= self.lr * vt
                else:
                    # Simple SGD update (No momentum)
                    x -= self.lr * dx
