# -*- coding: utf-8 -*-
"""Script to test the implementation"""

from train import *
from data import get_train_test_data, current_milli_time
from plot import *
from validation import select_best_hyper
import argparse
import sys
import random
import torch

torch.manual_seed(42)


def test_selected_model(
    activation: str, eta: float, momentum: float, plots: bool, n_runs: int
):
    """
    Test our implementation with the selected activation function and learning parameters, eventually plotting
    the learning curves and final result.
    :param activation: activation function to be used in the network
    :param eta: SGD learning rate
    :param momentum: SGD momentum factor
    :param plots: whether produce plots as in the report
    :param n_runs: number of runs for performance estimation
    """

    # Selecting a random model to plot the xy-axis, use random seed to select a different dataset at each run
    torch.manual_seed(current_milli_time())
    plot_model = random.randint(0, n_runs-1)
    torch.manual_seed(42)

    tot_loss = []
    tot_err = []
    tot_err_train = []
    tot_err_test = []
    epochs = 75
    batch_size = 100

    # Do the training
    print("Starting training on our implementation over {} runs".format(n_runs))
    train_data_full, train_targets_full, test_data_full, test_targets_full = get_train_test_data(1000, False, n_runs)
    for i in range(n_runs):
        # Get data and create selected network
        train_data, train_targets, test_data, test_targets = train_data_full[i], train_targets_full[i], \
                                                             test_data_full[i], test_targets_full[i]
        print("Building model {}...".format(i))
        if activation == "relu":
            model = myNN.Sequential(
                myNN.Linear(2, 25),
                myNN.ReLU(),
                myNN.Linear(25, 25),
                myNN.ReLU(),
                myNN.Linear(25, 25),
                myNN.ReLU(),
                myNN.Linear(25, 2),
            )
        else:
            model = myNN.Sequential(
                myNN.Linear(2, 25),
                myNN.Tanh(),
                myNN.Linear(25, 25),
                myNN.Tanh(),
                myNN.Linear(25, 25),
                myNN.Tanh(),
                myNN.Linear(25, 2),
            )

        # Train the network, Produce xy plots if required
        if plots and i == plot_model:
            print(
                "You chose to produce prediction visualization on a xy grid every 50 epochs.\n"
                "Model {} was randomly chosen for such plots. This train will require more time...".format(
                    i
                )
            )
            losses, errors = train_myNN(
                model,
                train_data,
                train_targets,
                test_data,
                test_targets,
                epochs,
                batch_size,
                eta,
                momentum,
                plots,
                activation
            )
        else:
            losses, errors = train_myNN(
                model,
                train_data,
                train_targets,
                test_data,
                test_targets,
                epochs,
                batch_size,
                eta,
                momentum,
                False,
            )

        # Add loss to list of loss
        tot_loss.append(losses)
        tot_err.append(errors)

        print(
            "Training on model {} finished, computing accuracy on train and test...".format(
                i
            )
        )

        # Compute error for train and test
        train_err = compute_errors(model, train_data, train_targets, batch_size)
        test_err = compute_errors(model, test_data, test_targets, batch_size)
        tot_err_train.append(train_err)
        tot_err_test.append(test_err)

        del model

    if plots:
        # Creating plots and saving them to pdf files
        print("-------------------------------------------------------")
        print("Saving requested plots for loss and errors")
        loss_save = "losstot_{act}_{n}runs".format(act=activation, n=n_runs)
        err_save = "err_{act}_{n}runs".format(act=activation, n=n_runs)
        plot_over_epochs(tot_loss, epochs, "Loss", loss_save)
        plot_over_epochs(tot_err, epochs, "Errors", err_save)

    # Computing and printing mean loss at each epochs over the runs
    mean_train = torch.mean(torch.Tensor([val["train"] for val in tot_loss]), 0)
    mean_test = torch.mean(torch.Tensor([val["test"] for val in tot_loss]), 0)
    for e in range(epochs):
        print(
            "Epoch {}, average train loss: {}, average test loss: {}".format(
                e, mean_train[e], mean_test[e]
            )
        )

    # Computing mean accuracy, std and mean train time over the runs
    mean_err_train = torch.mean(torch.Tensor(tot_err_train))
    mean_err_test = torch.mean(torch.Tensor(tot_err_test))
    var_err_train = torch.std(torch.Tensor(tot_err_train))
    var_err_test = torch.std(torch.Tensor(tot_err_test))
    print("-------------------------------------------------------")
    print("Final error count and standard deviation on train and test:")
    print(
        "Train -> Mean Error = {}, Standard deviation = {}".format(
            mean_err_train, var_err_train
        )
    )
    print(
        "Test -> Mean Error = {}, Standard deviation = {}".format(
            mean_err_test, var_err_test
        )
    )

    return


def test_pytorch_model(
    activation: str, eta: float, momentum: float, plots: bool, n_runs: int
):
    """
    Test pytorch implementation with the selected activation function and learning parameters, eventually plotting
    the learning curves.
    :param activation: activation function to be used in the network
    :param eta: SGD learning rate
    :param momentum: SGD momentum factor
    :param plots: whether produce plots as in the report
    :param n_runs: number of runs for performance estimation
    """

    tot_loss = []
    tot_err = []
    tot_err_train = []
    tot_err_test = []
    epochs = 75
    batch_size = 100

    # Test pytorch over n_runs
    print("Starting training on pytorch implementation over {} runs".format(n_runs))
    train_data_full, train_targets_full, test_data_full, test_targets_full = get_train_test_data(1000, False, n_runs)
    for i in range(n_runs):
        train_data, train_targets, test_data, test_targets = train_data_full[i], train_targets_full[i], \
                                                             test_data_full[i], test_targets_full[i]
        print("Building model {}...".format(i))
        if activation == "relu":
            model = nn.Sequential(
                nn.Linear(2, 25),
                nn.ReLU(),
                nn.Linear(25, 25),
                nn.ReLU(),
                nn.Linear(25, 25),
                nn.ReLU(),
                nn.Linear(25, 2),
            )
        else:
            model = nn.Sequential(
                nn.Linear(2, 25),
                nn.Tanh(),
                nn.Linear(25, 25),
                nn.Tanh(),
                nn.Linear(25, 25),
                nn.Tanh(),
                nn.Linear(25, 2),
            )

        # Train pytorch and record run losses and errors
        losses, errors = train_pytorch(
            model,
            train_data,
            train_targets,
            test_data,
            test_targets,
            epochs,
            batch_size,
            eta,
            momentum,
        )

        tot_loss.append(losses)
        tot_err.append(errors)

        print(
            "Training on model {} finished, computing accuracy on train and test...".format(
                i
            )
        )

        train_err = compute_errors(model, train_data, train_targets, batch_size)
        test_err = compute_errors(model, test_data, test_targets, batch_size)
        tot_err_train.append(train_err)
        tot_err_test.append(test_err)

        del model

    if plots:
        # Creating plots and saving them to pdf files
        print("-------------------------------------------------------")
        print("Saving requested plots for loss and errors")
        loss_save = "losstot_pytorch_{act}_{n}runs".format(act=activation, n=n_runs)
        err_save = "err_pytorch_{act}_{n}runs".format(act=activation, n=n_runs)
        plot_over_epochs(tot_loss, epochs, "Pytorch Loss", loss_save)
        plot_over_epochs(tot_err, epochs, "Pytorch Errors", err_save)

    # Computing and printing mean loss at each epochs over the runs
    mean_train = torch.mean(torch.Tensor([val["train"] for val in tot_loss]), 0)
    mean_test = torch.mean(torch.Tensor([val["test"] for val in tot_loss]), 0)
    for e in range(epochs):
        print(
            "Epoch {}, average train loss: {}, average test loss: {}".format(
                e, mean_train[e], mean_test[e]
            )
        )

    # Computing mean accuracy, std and mean train time over the runs
    mean_err_train = torch.mean(torch.Tensor(tot_err_train))
    mean_err_test = torch.mean(torch.Tensor(tot_err_test))
    var_err_train = torch.std(torch.Tensor(tot_err_train))
    var_err_test = torch.std(torch.Tensor(tot_err_test))
    print("-------------------------------------------------------")
    print(
        "Final error count and standard deviation on train and test for pytorch implementation:"
    )
    print(
        "Train -> Mean Error = {}, Standard deviation = {}".format(
            mean_err_train, var_err_train
        )
    )
    print(
        "Test -> Mean Error = {}, Standard deviation = {}".format(
            mean_err_test, var_err_test
        )
    )

    return


def main(activation: str, validation: bool, pytorch: bool, plots: bool, n_runs: int):
    """
    Run the selected implementation
    :param activation: ReLU or Tanh as activation
    :param validation: perform validation to find best hyper params
    :param pytorch: compare our performance with pytorch
    :param plots: produce plots as in the report
    :param n_runs: number of runs to estimate performances
    """
    torch.set_grad_enabled(False)

    # If no arguments are received, print help info
    if len(sys.argv) == 1:
        print("\n-------------------------------------------------------")
        print(
            "No arguments defined. Default configuration used.\n"
            "To receive help on how to set parameters \nand change configuration\n"
            "use the command: python test.py -h"
        )
        print("-------------------------------------------------------")

    # Define activation function
    if activation == "relu":
        print("ReLU activation function chosen.")
    else:
        print("Tanh activation function chosen.")

    print("-------------------------------------------------------")

    # Load best params, by computing them or already defined
    best_etas = {"relu": 0.1, "tanh": 0.1}
    best_momentum = {"relu": 0.6, "tanh": 0.9}
    if validation:
        print(
            "Starting validation algorithm on eta parameter for the chosen model. "
            "This may require a few hours"
        )
        # Fine grained search, coarse grained already done before
        etas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        momentums = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        best_param = select_best_hyper(activation, etas, momentums)
        eta = best_param["eta"]
        momentum = best_param["momentum"]
    else:
        print("Loading best eta parameter for the chosen model")
        eta = best_etas[activation]
        momentum = best_momentum[activation]

    print("-------------------------------------------------------")

    # Test our implementation
    test_selected_model(activation, eta, momentum, plots, n_runs)

    print("-------------------------------------------------------")

    # If selected, compare with pytorch
    if pytorch:
        torch.set_grad_enabled(True)
        print(
            "You chose to compare performances with Pytorch. A copy of the model with NN will be created"
        )
        test_pytorch_model(activation, eta, momentum, plots, n_runs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the DL Project 2 implementation.")
    # Loading model params

    title_activations = parser.add_argument_group(
        "Possible activations", "Select one of the two possible activations"
    )
    group_models = title_activations.add_mutually_exclusive_group()
    group_models.add_argument(
        "-tanh",
        action="store_const",
        help="Use Tanh as activation function (default)",
        dest="activation",
        const="tanh",
    )
    group_models.add_argument(
        "-relu",
        action="store_const",
        help="Use ReLU as activation function",
        dest="activation",
        const="relu",
    )

    parser.add_argument(
        "-validation",
        action="store_true",
        help="Run validation on the model. "
        "If not set, already selected best eta SGD param will be used.",
    )

    parser.add_argument(
        "-pytorch",
        action="store_true",
        help="Create a copy of the model in pytorch to estimate performances differences"
        "between our implementation and one using NN package (default false)",
    )

    parser.add_argument(
        "-plots",
        action="store_true",
        help="Create the errors/loss plot over epochs for the selected model as shown in the report. "
        "Create prediction visualization on a xy grid every 50 epochs",
    )

    parser.add_argument(
        "-n_runs",
        help="Define number of runs of the train/test process ",
        type=int,
        default=10,
    )

    parser.set_defaults(activation="tanh")
    args = parser.parse_args()

    main(args.activation, args.validation, args.pytorch, args.plots, args.n_runs)
