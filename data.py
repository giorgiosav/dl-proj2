# -*- coding: utf-8 -*-
"""Utility to get datapoints"""

import torch
import math


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


def get_train_test_data(n_points: int) -> tuple:
    """
    :param n_points: number of points to generate
    :return: train and test data + targets
    """

    train_data, train_targets = _generate_set(n_points)
    test_data, test_targets = _generate_set(n_points)

    return train_data, train_targets, test_data, test_targets
