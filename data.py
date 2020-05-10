# -*- coding: utf-8 -*-
"""Utility to get datapoints"""

import torch
import math
import time

# Used to set new seed at each data generation, while seed is fixed for model generation
current_milli_time = lambda: int(round(time.time() * 1000))

def _generate_set(n_points: int) -> tuple:
    """
    Generates 2D points randomly in [0,1]^2 and labels them with 1 if they are within
    a circle centered at [0.5, 0.5] with radius 1/sqrt(2*pi), else 0.
    :param n_points: number of points to generate
    :return: a tensor of point coordinates and a tensor of labels
    """

    # Set radius and center
    radius2 = 1 / (2 * math.pi)
    center = 0.5

    # Create points uniformly in [0, 1]
    points = torch.rand(n_points, 2)
    sums = points.sub(center).pow(2).sum(1)  # (x-0.5)^2 + (y-0.5)^2

    # Compute points labels
    labels = torch.where(
        sums <= radius2, torch.ones(n_points), torch.zeros(n_points)
    ).long()

    return points, labels


def get_train_test_data(n_points: int = 1000, random: bool = False, n_runs: int = 10) -> tuple:
    """
    :param random: to get a fully random dataset (for validation)
    :param n_runs: to get n_runs dataset with seeds fixed (for reproducibility)
    :param n_points: number of points to generate
    :return: train and test data + targets (single or n_runs of them)
    """

    if random:
        torch.manual_seed(current_milli_time())
        train_data, train_targets = _generate_set(n_points)
        torch.manual_seed(current_milli_time())
        test_data, test_targets = _generate_set(n_points)
    else:
        train_data, train_targets, test_data, test_targets = [], [], [], []
        for i in range(0, n_runs*2, 2):
            torch.manual_seed(i)
            train_data_i, train_targets_i = _generate_set(n_points)
            torch.manual_seed(i+1)
            test_data_i, test_targets_i = _generate_set(n_points)
            train_data.append(train_data_i)
            train_targets.append(train_targets_i)
            test_data.append(test_data_i)
            test_targets.append(test_targets_i)

    torch.manual_seed(42)

    return train_data, train_targets, test_data, test_targets
