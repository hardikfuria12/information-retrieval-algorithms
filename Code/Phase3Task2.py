# TODO Ari: Add SVD based clustering

import numpy as np
import pandas as pd
import random
from scipy.spatial.distance import cosine
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds

# modified from https://pythonprogramming.net/k-means-from-scratch-machine-learning-tutorial/
class K_means:
    def __init__(self, k=2, tol=.00001, max_iter=1000):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):

        self.centroids = {}
        self.labels = [0] * len(data)

        init_centroid_indices = random.sample(range(len(data)), self.k)
        print(init_centroid_indices)
        for i in range(self.k):
            # Initializes centroids to be first k points...
            self.centroids[i] = data[init_centroid_indices[i]]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for j, featureset in enumerate(data):
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))

                self.labels[j] = classification
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True
            # Check if optimized (at tol)
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
                    print(np.sum((current_centroid - original_centroid) / original_centroid * 100.0))
                    optimized = False
            if optimized:
                break

def is_undirected(G):
    # WARNING: MODIFIES IN-PLACE OUT OF SCOPe
    G[np.isnan(G)] = -1
    return np.allclose(G, G.T)

def build_undirected(G):
    # G is an adjacency matrix represented as a pandas dataframe
    G[np.isnan(G)] = -1
    bool_array = ~np.isclose(np.tril(G).T,np.triu(G))
    indices_to_fill = np.where(bool_array)
    for i, j in zip(indices_to_fill[0], indices_to_fill[1]):
        if G.iloc[i, j] < G.iloc[j, i]:
            G.iloc[i, j] = G.iloc[j, i]
        else:
            G.iloc[j, i] = G.iloc[i, j]
    return G


def compute_laplacian(G):
    # computes unnormalized laplacian
    W = G.values
    W[W<0] = 0
    D = np.diag(G.gt(0).sum(axis=1))

    return D - W


def spectral_cluster(L, k):
    # Clusters the laplacian matrix

    L_sparse = csc_matrix(L)
    _,_,Vt = svds(L_sparse, k=k)
    _, _, Vt = np.linalg.svd(L)
    # Takes smallest eigenvectors puts in matrix U dimension n x k
    U = Vt[-k:].T
    clf = K_means(k=k, tol=1e-15)
    clf.fit(U)

    return pd.Series(clf.labels)


def find_location_dir(imageid, ids, visualDict):
    '''
    Params: string image id, list of locations in underscore format
    Returns string location
    '''
    for location in ids:
        if imageid in visualDict[location]['CM'].keys():
            return location


def visualize(G, labels, k, cluster_method, ids, visualDict):
    # write some html files lol
    print(labels.value_counts())

    for i in range(int(k)):
        with open(f'cluster{i+1}.html', 'w') as handle:
            content = """<html><head></head><body>"""
            content += "<table style=\"border: 2px solid black; margin:10px;\"><tr>"
            content += f"<th style=\"border: 2px solid black; margin:10px;\">Cluster {i+1}</th>"
            content += "</tr><tr>"
            if cluster_method == '1':
                cluster_id = G.index[labels[labels == i].index]
            else: # svd cluster
                cluster_id = G.index[labels[labels == i].index]
            content += "<td style=\"border: 2px solid black; margin:10px;vertical-align: top;\">"

            # Add the images
            for imageid in cluster_id:
                directory = find_location_dir(imageid, ids, visualDict)
                file = imageid
                content += "<img src=\"" + f'../img/{directory}/{file}.jpg' + "\"style=\"height:30%; width:30%;\"><br>"

            # close the cell
            content += "</td>"
            content += "</tr>"
            content += "</table>"
            content += """</body></html>"""

            handle.write(content)


def svd_cluster(G,k):
    G_sparse = csc_matrix(G.values)
    _,_,Vt = svds(G_sparse, k=k+1)

    leaders = Vt[:k+1]
    labels = []
    for index, row in G.iterrows():
        min_dist = float('inf')
        label = -1

        for k, leader in enumerate(leaders):
            dist = cosine(row, leader)
            if dist < min_dist:
                min_dist = dist
                label = k
        labels.append(label)
    return pd.Series(labels)


def main(imgAdjMatrix, ids, visualDict, k, cluster_method):
    # build an undirected graph from the imgAdjMatrix
    G = build_undirected(imgAdjMatrix)
    if cluster_method == '1': # spectral
        L = compute_laplacian(G)
        labels = spectral_cluster(L, int(k))
    else: # svd based
        labels = svd_cluster(G,int(k))

    visualize(G, labels, k, cluster_method, ids, visualDict)