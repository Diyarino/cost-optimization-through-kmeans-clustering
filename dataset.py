# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 09:01:24 2025

@author: Diyar Altinses, M.Sc.
"""

# %% imports

import torch

from sklearn.datasets import make_blobs

# %% 

def generate_cost_drivers(n_samples=500, n_features=2, centers=4):
    X, y = make_blobs(n_samples=n_samples, 
                      n_features=n_features, 
                      centers=centers, 
                      cluster_std=1.2,
                      random_state=42)
    return torch.tensor(X, dtype=torch.float32), y

# %% run

if __name__ == '__main__':
    data = generate_cost_drivers()