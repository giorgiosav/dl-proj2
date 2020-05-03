# -*- coding: utf-8 -*-
"""Plotting utility"""

import matplotlib.pyplot as plt
import matplotlib
import torch
import math

# Used to save in LaTeX design
# This configuration has been commented to make the implementation work in the VM.
# The plot are reproducible even without it, but they won't have the "LaTeX style"
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})


def visualize_predictions(data, target, epoch, test_label, savename):
    colors = ["orangered", "blue"]
    fig = plt.figure(figsize=(10, 10))
    circle = plt.Circle((0.5, 0.5), 1 / math.sqrt(2 * math.pi), color='black', alpha=0.075)
    ax = fig.gca()
    ax.add_artist(circle)
    ax.scatter(data[:, 0], data[:, 1], c=target, cmap=matplotlib.colors.ListedColormap(colors), s=20)
    plt.xlabel(r'\textbf{x}', fontsize=11)
    plt.ylabel(r'\textbf{y}', fontsize=11)
    plt.title(r'\textbf{Data classes ' + test_label + ' - Epoch ' + str(epoch) + '}', fontsize=15)
    plt.savefig("plot/circles/" + savename + ".pdf")
    plt.close()


def plot_over_epochs(values_list: list, epochs: int, label: str, savename: str):
    """
    Plots values vs epochs and save figure
    :param values_list: list of dict to plot (keys are train and test)
    :param epochs: number of epochs
    :param label: y axis label
    :param savename: output file name
    """

    # Compute the average of the value to plot,
    # together with maximum down/up displacement from the average at each epoch
    mean_train = torch.mean(torch.Tensor([val['train'] for val in values_list]), 0)
    mean_test = torch.mean(torch.Tensor([val['test'] for val in values_list]), 0)
    err_up_train = torch.max(torch.Tensor([val['train'] for val in values_list]), 0)[0] - mean_train
    err_down_train = mean_train - torch.min(torch.Tensor([val['train'] for val in values_list]), 0)[0]
    err_up_test = torch.max(torch.Tensor([val['test'] for val in values_list]), 0)[0] - mean_test
    err_down_test = mean_test - torch.min(torch.Tensor([val['test'] for val in values_list]), 0)[0]
    epochs_range = range(0, epochs)

    plt.figure()

    # Plot data and save figure
    err_train = [err_down_train, err_up_train]
    err_test = [err_down_test, err_up_test]
    markers, caps, bars = plt.errorbar(epochs_range, mean_train, yerr=err_train, label="Train " + label,
                                       color="blue")
    [bar.set_alpha(0.5) for bar in bars]
    markers, caps, bars = plt.errorbar(epochs_range, mean_test, yerr=err_test, label="Test " + label,
                                       color="orange")
    [bar.set_alpha(0.5) for bar in bars]
    plt.xticks(range(0, epochs, 2))

    # set labels (LaTeX can be used) -> Note: with the setting deactivated, this will print \textbf{...}
    plt.xlabel(r'\textbf{Epochs}', fontsize=11)
    plt.ylabel(r'\textbf{' + label + '}', fontsize=11)
    plt.legend()
    plt.savefig("plot/" + savename + ".pdf")
    plt.close()
