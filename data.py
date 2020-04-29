import torch
import math


def _generate_set(n_points):
    
    radius2 = 1 / (2 * math.pi)
    center = 0.5

    points = torch.rand(n_points, 2)
    sums = points.sub(center).pow(2).sum(1) # (x-0.5)^2 + (y-0.5)^2

    data = torch.where(sums <= radius2, torch.ones(n_points), torch.zeros(n_points))

    return data
    

def get_train_test_data(n_points):

    return _generate_set(n_points), _generate_set(n_points)
