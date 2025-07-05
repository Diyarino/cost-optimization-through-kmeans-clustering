# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 09:04:20 2025

@author: Altinses
"""

# %% imports

import torch

# %% Kmeans

class KMeans:
    def __init__(self, n_clusters=4, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.history = {'centroids': [], 'assignments': []}
        
    def fit(self, X):
        indices = torch.randint(0, len(X), (1,))
        centroids = X[indices]
        
        for _ in range(1, self.n_clusters):
            distances = torch.cdist(X, centroids).min(dim=1)[0]
            probabilities = distances / distances.sum()
            new_idx = torch.multinomial(probabilities, 1)
            centroids = torch.cat([centroids, X[new_idx]])
        
        for iteration in range(self.max_iter):
            distances = torch.cdist(X, centroids)
            assignments = torch.argmin(distances, dim=1)
            
            new_centroids = torch.stack([
                X[assignments == k].mean(dim=0) 
                for k in range(self.n_clusters)
            ])
            
            self.history['centroids'].append(centroids.clone())
            self.history['assignments'].append(assignments.clone())
            
            shift = torch.norm(new_centroids - centroids)
            if shift < self.tol:
                break
                
            centroids = new_centroids
            
        self.centroids = centroids
        return self
    
    def predict(self, X):
        distances = torch.cdist(X, self.centroids)
        return torch.argmin(distances, dim=1)
    
# %% run

if __name__ == '__main__':
    model = KMeans()
