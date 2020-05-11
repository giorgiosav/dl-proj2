# -*- coding: utf-8 -*-
"""Train algorithms"""

import torch
from torch import nn
import myNN
import myOptim
from typing import Union
from plot import visualize_predictions


def one_hot(tensor: torch.Tensor) -> torch.Tensor:
    """
    :param tensor: input tensor of targets, of shape Nx2
    :return: one-hot representation of input tensor
    """
    return torch.zeros(tensor.shape[0], 2).scatter_(1, tensor.view(-1, 1), 1.0)


def train_myNN(
    model: myNN.Sequential,
    train_data: torch.Tensor,
    train_targets: torch.Tensor,
    test_data: torch.Tensor,
    test_targets: torch.Tensor,
    epochs: int,
    batch_size: int,
    eta: float,
    momentum: float,
    plots: bool = False,
    activation: str = "",
    verbose: bool = False,
) -> tuple:
    """
    Train our implementation and record losses at each epoch (for train and test), eventually producing plots
    :param model: the model implemented using myNN package
    :param train_data: train datapoints
    :param train_targets: target for training
    :param test_data: test datapoints
    :param test_targets: target for testing
    :param epochs: number of epochs
    :param batch_size: size of each batch
    :param eta: SGD learning rate
    :param momentum: SGD momentum factor
    :param plots: produce intermediate plots or not
    :param activation: activation function (just to record it in the plot name)
    :param verbose: print logging info
    :return: (losses, errors): dictionaries with loss and number of errors at each step, for train and test
    """

    # Encode as onehot for training
    train_targets_onehot = one_hot(train_targets)
    test_targets_onehot = one_hot(test_targets)

    # Define MSE and SGD
    criterion = myNN.LossMSE()
    optimizer = myOptim.SGD(model.param(), eta, momentum)

    losses = {"train": [], "test": []}
    errors = {"train": [], "test": []}

    # Train
    for e in range(epochs):
        loss_acc = 0

        for data, target in zip(train_data.split(batch_size), train_targets_onehot.split(batch_size)):
            # Compute prediction, loss and go in backward pass
            prediction = model(data)
            loss = criterion(prediction, target)
            loss_acc += loss

            # Zero gradient and do a step
            model.zero_grad()
            model.backward(criterion.backward())
            optimizer.step()

        # Record loss/errors for train and test at a given epoch
        if verbose:
            print("Epoch: {}, avg loss per batch: {}".format(e, loss_acc / batch_size))
        losses["train"].append(loss_acc / batch_size)
        loss_test = _compute_loss_test(model, test_data, test_targets_onehot, batch_size, criterion, verbose)
        losses["test"].append(loss_test)
        errors["train"].append(compute_errors(model, train_data, train_targets, batch_size))
        errors["test"].append(compute_errors(model, test_data, test_targets, batch_size))

        # If required, save intermediate xy plot
        if plots and (e % 50 == 0 or e == epochs - 1):
            if verbose:
                print("Saving xy-plot for epoch {}".format(e))
            classes = _prepare_plot_data(model, test_data, batch_size)
            visualize_predictions(test_data, classes, e, "test", "test_classes_" + activation + str(e))

    return losses, errors


def train_pytorch(
    model: nn.Sequential,
    train_data: torch.Tensor,
    train_targets: torch.Tensor,
    test_data: torch.Tensor,
    test_targets: torch.Tensor,
    epochs: int,
    batch_size: int,
    eta: float,
    momentum: float,
    verbose: bool = False,
) -> tuple:
    """
    Train pytorch implementation and record losses at each epoch (for train and test)
    :param model: the model implemented using torch.nn package
    :param train_data: train datapoints
    :param train_targets: target for training
    :param test_data: test datapoints
    :param test_targets: target for testing
    :param epochs: number of epochs
    :param batch_size: size of each batch
    :param eta: SGD learning rate
    :param momentum: SGD momentum factor
    :param verbose: print logging info
    :return: (losses, errors): dictionaries with loss and number of errors at each step, for train and test
    """

    # Convert to onehot tensor to use for MSE
    train_targets_onehot = one_hot(train_targets)
    test_targets_onehot = one_hot(test_targets)

    # Define criterion and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=eta, momentum=momentum)

    losses = {"train": [], "test": []}
    errors = {"train": [], "test": []}

    for e in range(epochs):
        loss_acc = 0

        # Train over a batch
        for data, target in zip(train_data.split(batch_size), train_targets_onehot.split(batch_size)):
            prediction = model(data)
            loss = criterion(prediction, target)
            loss_acc += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if verbose:
            print("Epoch: {}, avg loss per batch: {}".format(e, loss_acc / batch_size))

        # Record the loss and errors
        with torch.no_grad():
            losses["train"].append(loss_acc / batch_size)
            loss_test = _compute_loss_test(model, test_data, test_targets_onehot, batch_size, criterion, verbose)
            losses["test"].append(loss_test)
            errors["train"].append(compute_errors(model, train_data, train_targets, batch_size))
            errors["test"].append(compute_errors(model, test_data, test_targets, batch_size))

    return losses, errors


def compute_errors(model: Union[myNN.Sequential, nn.Sequential], data: torch.Tensor, targets: torch.Tensor, batch_size: int,) -> int:
    """
    Compute number of errors for the given model, and datapoints
    :param model: DL model implemented with pytorch or myNN
    :param data: datapoints
    :param targets: target for datapoints
    :param batch_size: size of batch
    :return: number of errors for the given dataset
    """
    tot_err = 0

    # Get classes for each batch and record error numbers for each batch
    for inp, targ in zip(data.split(batch_size), targets.split(batch_size)):
        classes = _compute_classes(model, inp)
        # Batch_errors = Batch_size - correct_predictions
        tot_err += classes.shape[0] - torch.sum(classes == targ).item()

    return tot_err


def _compute_classes(model: Union[myNN.Sequential, nn.Sequential], inp: torch.Tensor) -> torch.Tensor:
    """
    Compute the classes from the network output (the index of the max in each tensor is the predicted class)
    :param model: DL model implemented with pytorch or myNN
    :param inp: single or batch of inputs
    :return: predicted class or classes for the input or batch of inputs
    """

    # Predict and get max index
    prediction = model(inp)
    classes = prediction.max(1)[1]
    return classes


def _prepare_plot_data(model: Union[myNN.Sequential, nn.Sequential], data: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Utility to produce classes for the whole dataset, ready to be plot
    :param model: DL model implemented with pytorch or myNN
    :param data: datapoints
    :param batch_size: size of each batch
    :return:
    """

    # Predict, get class of max index and concatenate results
    tot_class = []
    for inp in data.split(batch_size):
        classes = _compute_classes(model, inp)
        tot_class.append(classes)

    return torch.cat(tot_class, dim=0)


def _compute_loss_test(
    model: Union[myNN.Sequential, nn.Sequential],
    test_data: torch.Tensor,
    test_targets: torch.Tensor,
    batch_size: int,
    criterion: myNN.LossMSE,
    verbose: bool = False,
):
    """
    :param model: the model implemented using torch.nn package
    :param test_data: test datapoints
    :param test_targets: target for testing
    :param batch_size: batch dimension
    :param criterion: MSE criterion
    :param verbose: logging info
    :return:
    """

    # No grad added to use the function for both pytorch and myNN models
    with torch.no_grad():
        # Compute loss over the batch
        loss_acc = 0
        for data, target in zip(test_data.split(batch_size), test_targets.split(batch_size)):
            prediction = model(data)
            loss = criterion(prediction, target)
            loss_acc += loss

        # Return avg loss per batch
        if verbose:
            print("avg loss test per batch: {}".format(loss_acc / batch_size))

        return loss_acc / batch_size
