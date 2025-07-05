# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 09:00:34 2025

@author: Diyar Altinses, M.Sc.
"""

# %%

import os
from PIL import Image
import matplotlib.pyplot as plt

from dataset import generate_cost_drivers
from kmeans import KMeans
from evaluate import evaluate_clustering
from config_plots import configure_plt

# %%

def main():
    configure_plt()
    X, true_labels = generate_cost_drivers(centers=7)
    kmeans = KMeans(n_clusters=6)
    kmeans.fit(X)
    y_pred = kmeans.predict(X)
    evaluate_clustering(X, y_pred, kmeans.centroids, true_labels)
    
    print("\Create Animations-GIF...")
    
    def update(frame):
        fig, ax = plt.subplots(figsize=(6, 3))
        colors = ['#4EACC5', '#FF9C34', '#4E9A06', '#FF3300', '#9370DB', 
                  'yellow', 'gray', 'midnightblue']
        centroids = kmeans.history['centroids'][frame]
        assignments = kmeans.history['assignments'][frame]
        
        for k in range(len(centroids)):
            cluster_data = X[assignments == k]
            ax.scatter(cluster_data[:, 0], cluster_data[:, 1], 
                      c=colors[k], marker='o', label=f'Cluster {k}', alpha=0.6)
        ax.scatter(centroids[:, 0], centroids[:, 1], 
                  c='black', marker='X', s=200, label='Centroids')
        ax.set_title(f'k-Means Iteration {frame+1}')
        ax.set_xlabel('Cost driver 1')
        ax.set_ylabel('Cost driver 2')
        plt.grid()
        plt.tight_layout()
        # ax.legend()
    
    temp_dir = "kmeans_frames"
    os.makedirs(temp_dir, exist_ok=True)
    
    for frame in range(len(kmeans.history['centroids'])):
        update(frame)
        plt.savefig(f"{temp_dir}/frame_{frame:03d}.png", dpi = 150, bbox_inches='tight')
        
    
    frames = []
    for frame in range(len(kmeans.history['centroids'])):
        img = Image.open(f"{temp_dir}/frame_{frame:03d}.png")
        frames.append(img)
    
    frames[0].save("kmeans_clustering.gif",
                   format="GIF",
                   append_images=frames[1:],
                   save_all=True,
                   duration=1000,
                   loop=0)
    
    # Aufräumen
    for frame in range(len(kmeans.history['centroids'])):
        os.remove(f"{temp_dir}/frame_{frame:03d}.png")
    os.rmdir(temp_dir)
    
    print("Animation created")

if __name__ == "__main__":
    main()