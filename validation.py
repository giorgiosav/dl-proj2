# -*- coding: utf-8 -*-
"""Validation algorithms to compute best hyperparameters"""

from train import train_myNN, compute_errors
import sys
import myNN
from data import get_train_test_data


def select_best_hyper(activation: str, etas: list, momentums: list, n_runs: int = 5,
                      epochs: int = 50, batch_size: int = 100, verbose: bool = True) -> dict:
    best_err = sys.float_info.max
    best_params = {"eta": 0, "momentum": 0}

    for eta in etas:
        for momentum in momentums:
            tot_err = 0
            for i in range(0, n_runs):
                # Create net, train it and compute accuracy on test data
                if activation == "relu":
                    model = myNN.Sequential(
                        myNN.Linear(2, 25),
                        myNN.ReLU(),
                        myNN.Linear(25, 25),
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
                        myNN.Linear(25, 25),
                        myNN.Tanh(),
                        myNN.Linear(25, 2),
                    )
                criterion = myNN.LossMSE()
                # A new train/test set is used at each run to avoid overfitting a dataset
                train_data, train_targets, test_data, test_targets = get_train_test_data(1000)
                train_myNN(model, train_data, train_targets, test_data, test_targets,
                           epochs, batch_size, eta, momentum)
                err = compute_errors(model, test_data, test_targets, batch_size)
                tot_err += err
                del model
            err_run = tot_err / n_runs
            # Save accuracy if better than current best
            if verbose:
                print("Eta = {}, momentum = {}, avg_err = {}".
                      format(eta, momentum, err_run))
            if err_run < best_err:
                best_err = err_run
                best_params["eta"] = eta
                best_params["momentum"] = momentum
                if verbose:
                    print("New best combination: Eta = {}, momentum = {}, avg_err = {}".
                      format(eta, momentum, err_run))

    print("Best result found! Eta = {}, momentum = {}, avg_err = {}".
          format(best_params["eta"], best_params["momentum"], best_err))
    return best_params
