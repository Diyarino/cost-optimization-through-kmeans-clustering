# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 09:06:55 2025

@author: Diyar Altinses, M.Sc.
"""

# %% imports

import matplotlib.pyplot as plt

# %% evaluate

def evaluate_clustering(X, y_pred, centroids, true_labels=None):
    plt.figure(figsize=(10, 3))
    
    colors = ['#4EACC5', '#FF9C34', '#4E9A06', '#FF3300', '#9370DB', 
              'yellow', 'gray', 'midnightblue']

    plt.subplot(1, 2, 1)
    for k in range(len(centroids)):
        cluster_data = X[y_pred == k]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                    c=colors[k], marker='o', alpha=0.6)
    plt.scatter(centroids[:, 0], centroids[:, 1], 
                c='black', marker='X', s=200, label='Centroids')
    plt.xlabel('Cost driver 1')
    plt.ylabel('Cost driver 2')
    plt.grid()
    # plt.legend()
    
    if true_labels is not None:
        from sklearn.metrics import silhouette_score
        sil_score = silhouette_score(X, y_pred)
        plt.subplot(1, 2, 2)
        plt.barh(['Silhouette Score'], [sil_score], color='#4EACC5')
        plt.xlim(0, 1)
    
    plt.tight_layout()
    plt.savefig('evaluate.png', dpi = 300, bbox_inches='tight')
    plt.show()
    
# %% run

if __name__ == '__main__':
    test = 0.0

    
    