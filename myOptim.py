import torch

class SGD:
    def __init__(self, params, lr):
        '''
        Python passes params by reference - just make sure not to reassign
        the params in your module
        '''
        self.params = params
        self.lr = lr

    def step(self):
        for param_group in self.params:
            for tup in param_group:
                x, dx = tup
                x -= self.lr * dx