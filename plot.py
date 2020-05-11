# -*- coding: utf-8 -*-
"""Plotting utility"""

import matplotlib.pyplot as plt
import matplotlib
import torch
import math

# Used to save in LaTeX design
# This configuration has been commented to make the implementation work in the VM.
# The plot are reproducible even without it, but they won't have the "LaTeX style"
# matplotlib.use("pgf")
# matplotlib.rcParams.update(
#     {
#         "pgf.texsystem": "pdflatex",
#         "font.family": "serif",
#         "text.usetex": True,
#         "pgf.rcfonts": False,
#     }
# )


def visualize_predictions(data: torch.Tensor, target: torch.Tensor, epoch: int, test_label: str, savename: str):
    """
    Plot prediction on a xy axis
    :param data: datapoints
    :param target: targets of datapoints
    :param epoch: epoch in training
    :param test_label: add a label indicating if the plot is for train or test
    :param savename: name used for saving
    """

    # plot a circle defining the area in which the points are labeled as "1"
    colors = ["orangered", "blue"]
    fig = plt.figure(figsize=(10, 10))
    circle = plt.Circle((0.5, 0.5), 1 / math.sqrt(2 * math.pi), color="black", alpha=0.075)
    ax = fig.gca()
    ax.add_artist(circle)

    # Plot points
    ax.scatter(
        data[:, 0], 
        data[:, 1], 
        c=target, 
        cmap=matplotlib.colors.ListedColormap(colors), 
        s=20,
    )

    # Set axis and label
    plt.xlabel(r"\textbf{x}", fontsize=11)
    plt.ylabel(r"\textbf{y}", fontsize=11)
    plt.title(r"\textbf{Data classes " + test_label + " - Epoch " + str(epoch) + "}", fontsize=15)
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
    mean_train = torch.mean(torch.Tensor([val["train"] for val in values_list]), 0).tolist()
    mean_test = torch.mean(torch.Tensor([val["test"] for val in values_list]), 0).tolist()
    epochs_range = range(0, epochs)

    plt.figure()

    # Plot data and save figure
    plt.plot(epochs_range, mean_train, label="Train " + label, color="blue")
    plt.plot(epochs_range, mean_test, label="Test " + label, color="orange")
    xticks = list(range(0, epochs, 25))
    xticks.append(epochs - 1)
    plt.xticks(xticks)
    plt.grid(linestyle="dotted")

    # set labels (LaTeX can be used) -> Note: with the setting deactivated, this will print \textbf{...}
    plt.xlabel(r"\textbf{Epochs}", fontsize=11)
    plt.ylabel(r"\textbf{" + label + "}", fontsize=11)
    plt.legend()
    plt.savefig("plot/" + savename + ".pdf")
    plt.close()
