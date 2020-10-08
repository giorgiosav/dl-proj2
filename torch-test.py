import torch
import myNN
import math
from torch.nn.parameter import Parameter


def assert_tensors_equal(t1, t2):
    s = torch.sum(t1 - t2)
    assert math.isclose(s, 0, abs_tol=1e-07)


def test_linear(nb_hidden1, nb_hidden2, batch_size):
    myLinear = myNN.Linear(nb_hidden1, nb_hidden2)
    torchLinear = torch.nn.Linear(nb_hidden1, nb_hidden2)

    in_tensor = torch.rand((batch_size, nb_hidden1))
    weights = torch.rand((nb_hidden2, nb_hidden1))
    biases = torch.rand(nb_hidden2)

    myLinear.weights = weights
    myLinear.bias = biases

    torchLinear.weight = Parameter(weights)
    torchLinear.bias = Parameter(biases)

    myl = myLinear(in_tensor)
    torchl = torchLinear(in_tensor)

    assert_tensors_equal(myl, torchl)


def test_paramless_module(myModule, torchModule, batch_size):
    tensor_size = 100
    in_tensor = torch.rand((batch_size, tensor_size)) * 50 - 25  # range -25, 25

    myt = myModule(in_tensor)
    torcht = torchModule(in_tensor)

    assert_tensors_equal(myt, torcht)


def test_mse(tensor_shape):
    myMSE = myNN.LossMSE()
    torchMSE = torch.nn.MSELoss()

    tensor_size = 100
    in_tensor = torch.rand(tensor_shape)
    target = torch.rand(tensor_shape)

    myl = myMSE(in_tensor, target)
    torchl = torchMSE(in_tensor, target)

    assert_tensors_equal(myl, torchl)


def main():
    myReLU = myNN.ReLU()
    myTanh = myNN.Tanh()

    torchReLU = torch.nn.ReLU()
    torchTanh = torch.nn.Tanh()

    test_linear(10, 50, 1)
    test_linear(10, 50, 10)

    test_paramless_module(myReLU, torchReLU, 1)
    test_paramless_module(myReLU, torchReLU, 10)

    test_paramless_module(myTanh, torchTanh, 1)
    test_paramless_module(myTanh, torchTanh, 10)

    test_mse(100)
    test_mse((10, 100))
    # test_mse((10, 100, 23, 42, 61))

    print("all tests passed!")


if __name__ == "__main__":
    main()
