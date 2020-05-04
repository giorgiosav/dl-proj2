import torch
import math


def _generate_set(n_points):
    """
    :param n_points: (int) number of points to generate
    :return: (tuple) a tensor of point coordinates and a tensor of labels

    Generates 2D points randomly in [0,1]^2 and labels them with 1 if they are within 
    a circle centered at [0.5, 0.5] with radius 1/sqrt(2*pi), else 0.
    """
    radius2 = 1 / (2 * math.pi)
    center = 0.5

    points = torch.rand(n_points, 2)
    sums = points.sub(center).pow(2).sum(1)  # (x-0.5)^2 + (y-0.5)^2

    labels = torch.where(
        sums <= radius2, torch.ones(n_points), torch.zeros(n_points)
    ).long()

    return points, labels


def get_train_test_data(n_points):
    """
    :param n_points: (int) number of points to generate
    :return: (tuple) train and test data + targets
    """
    train_data, train_targets = _generate_set(n_points)
    test_data, test_targets = _generate_set(n_points)

    return train_data, train_targets, test_data, test_targets
