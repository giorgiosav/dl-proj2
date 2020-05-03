import torch


class SGD:
    def __init__(self, params, lr, momentum=0):
        """
        Python passes params by reference - just make sure not to reassign
        the params in your module
        Note: momentum implementation is same as pytorch
        """
        self.params = params
        self.lr = lr
        self.momentum = momentum
        if self.momentum != 0:
            self.vt_prev = []
            for tuple in params:
                for _, dx in tuple:
                    self.vt_prev.append(dx)

    def step(self):
        for (tuple_index, tuple) in enumerate(self.params):
            for param_index, (x, dx) in enumerate(tuple):
                i = tuple_index + param_index
                if self.momentum != 0:
                    vt = self.vt_prev[i] * self.momentum + dx
                    self.vt_prev[i] = vt
                    x -= self.lr * vt
                else:
                    x -= self.lr * dx
