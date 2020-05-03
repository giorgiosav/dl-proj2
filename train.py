import torch
from torch import nn
import myNN
import myOptim
from typing import Union
from plot import visualize_predictions


def one_hot(tensor: torch.Tensor) -> torch.Tensor:
    return torch.zeros(tensor.shape[0], 2).scatter_(1, tensor.view(-1, 1), 1.0)


def train_myNN(model: myNN.Sequential,
               train_data: torch.Tensor, train_targets: torch.Tensor,
               test_data: torch.Tensor, test_targets: torch.Tensor,
               epochs: int, batch_size: int, eta: float, momentum: float,
               plots: bool = False, verbose: bool = False) -> tuple:
    train_targets_onehot = one_hot(train_targets)
    test_targets_onehot = one_hot(test_targets)

    criterion = myNN.LossMSE()
    optimizer = myOptim.SGD(model.param(), eta, momentum)

    losses = {"train": [], "test": []}
    errors = {"train": [], "test": []}

    for e in range(epochs):
        loss_acc = 0

        for data, target in zip(train_data.split(batch_size), train_targets_onehot.split(batch_size)):
            prediction = model(data)
            loss = criterion(prediction, target)
            loss_acc += loss

            model.zero_grad()
            model.backward(criterion.backward())  # TODO: add support for just criterion.backward()?
            optimizer.step()

        if verbose: print("Epoch: {}, avg loss per batch: {}".format(e, loss_acc / batch_size))
        losses["train"].append(loss_acc / batch_size)
        loss_test = _compute_loss_test(model, test_data, test_targets_onehot, batch_size, criterion, verbose)
        losses["test"].append(loss_test)
        errors["train"].append(compute_errors(model, train_data, train_targets, batch_size))
        errors["test"].append(compute_errors(model, test_data, test_targets, batch_size))

        if (e % 50 == 0 or e == epochs - 1) and plots:
            if verbose: print("Saving xy-plot for epoch {}".format(e))
            classes = _prepare_plot_data(model, test_data, batch_size)
            visualize_predictions(test_data, classes, e, "test", "test_classes" + str(e))

    return losses, errors


def train_pytorch(model: nn.Sequential,
                  train_data: torch.Tensor, train_targets: torch.Tensor,
                  test_data: torch.Tensor, test_targets: torch.Tensor,
                  epochs: int, batch_size: int, eta: float, momentum: float, verbose: bool = False) -> tuple:
    train_targets_onehot = one_hot(train_targets)
    test_targets_onehot = one_hot(test_targets)

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=eta, momentum=momentum)

    losses = {"train": [], "test": []}
    errors = {"train": [], "test": []}

    for e in range(epochs):
        loss_acc = 0

        for data, target in zip(train_data.split(batch_size), train_targets_onehot.split(batch_size)):
            prediction = model(data)
            loss = criterion(prediction, target)
            loss_acc += loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if verbose: print("Epoch: {}, avg loss per batch: {}".format(e, loss_acc / batch_size))
        with torch.no_grad():
            losses["train"].append(loss_acc / batch_size)
            loss_test = _compute_loss_test(model, test_data, test_targets_onehot, batch_size, criterion, verbose)
            losses["test"].append(loss_test)
            errors["train"].append(compute_errors(model, train_data, train_targets, batch_size))
            errors["test"].append(compute_errors(model, test_data, test_targets, batch_size))

    return losses, errors


def compute_errors(model: Union[myNN.Sequential, nn.Sequential], data: torch.Tensor,
                   targets: torch.Tensor, batch_size: int) -> int:
    tot_err = 0

    for inp, targ in zip(data.split(batch_size), targets.split(batch_size)):
        classes = _compute_classes(model, inp)
        tot_err += classes.shape[0] - torch.sum(classes == targ).item()

    return tot_err


def _compute_classes(model: Union[myNN.Sequential, nn.Sequential], inp: torch.Tensor) -> torch.Tensor:
    prediction = model(inp)
    classes = prediction.max(1)[1]
    return classes


def _prepare_plot_data(model: Union[myNN.Sequential, nn.Sequential],
                       data: torch.Tensor, batch_size: int) -> torch.Tensor:
    tot_class = []
    for inp in data.split(batch_size):
        classes = _compute_classes(model, inp)
        tot_class.append(classes)

    return torch.cat(tot_class, dim=0)


def _compute_loss_test(model: Union[myNN.Sequential, nn.Sequential],
                       test_data: torch.Tensor, test_targets: torch.Tensor,
                       batch_size: int, criterion: myNN.LossMSE, verbose: bool = False):
    with torch.no_grad():
        loss_acc = 0
        for data, target in zip(test_data.split(batch_size), test_targets.split(batch_size)):
            prediction = model(data)
            loss = criterion(prediction, target)
            loss_acc += loss

        if verbose: print("avg loss test per batch: {}".format(loss_acc / batch_size))
    return loss_acc / batch_size
