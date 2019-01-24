import numpy as np
import os
import pickle
import Phase3Task1
from scipy.sparse import csc_matrix
import json


# THE CONTROL BOARD
walk = 0.85
min_gain = 0.04
max_iters = 30
# 175 seems like a sweet spot
pics_to_show = 100

project_folder = ""
pickle_folder = ""


def page_rank(k, image_dict):
    # k=int(k) #changed by H
    image_similarity_matrix = _get_image_similarity_matrix(image_dict)
    image_similarity_matrix.fillna(value=0, inplace=True)
    image_similarity_matrix = image_similarity_matrix.values
    for i in range(0, len(image_similarity_matrix)):
        image_similarity_matrix[i][i] = 0
    page_ranks = _page_rank(image_similarity_matrix)
    _print_k(k, page_ranks, image_similarity_matrix, list(image_dict.keys()))


def _get_image_similarity_matrix(image_dict):
    pickle_file_name = "imgAdjMatrixTextual.pickle"
    pickle_file_location = "{0}{1}".format(pickle_folder, pickle_file_name)
    if os.path.isfile(pickle_file_location):
        pickle_file = open(pickle_file_location, 'rb')
        sim_mat = pickle.load(pickle_file)
        pickle_file.close()
    else:
        k, sim_mat = Phase3Task1.task34help(10, image_dict)
        pickle_file = open(pickle_file_location, 'wb')
        pickle.dump(sim_mat, pickle_file)
        pickle_file.close()
    return sim_mat


def _find_directory(image_id):
    path = ""
    for subdir, dirs, files in os.walk('..\img'):
        for file in files:
            if image_id in file:
                path = os.path.join(subdir, file)
                break
    return os.path.abspath(path)


def _create_nodes_and_edges_str(page_ranks, adj_matrix, image_id_array, k):
    # sample node:
    # {id: 1, label: '348579551 - 0.72 ', value: 0.72,
    #  image: 'file:///C:/Users/amd30/Downloads/cse515/img/cabrillo/348579551.jpg', shape: 'image'}
    # id: incrementing integer, label: imageId - page rank, image: location of image, shape: "image"

    # sample edge:
    # {from: 1, to: 3, label: "0.34"}
    # from = id of node from, to = id of node to, label = string of similarity

    if k >= len(page_ranks):
        indices_to_print = list(range(0, len(page_ranks)))
    else:
        indices_to_print = list(np.asarray(page_ranks).argsort()[-k:])

    nodes = []
    edges = []

    for i in range(0, len(indices_to_print)):
        k = indices_to_print[i]
        image_id = image_id_array[k]
        label = "{0} ● {1}".format(image_id, page_ranks[k])
        nodes.append({"id": int(k), "label": label, "value": page_ranks[k],
                      "image": "file:///{0}".format(_find_directory(image_id)), "shape": "image"})

        for j in range(0, len(indices_to_print)):  # I am making the assumption that the imgadjarr is square
            m = indices_to_print[j]
            if adj_matrix[k][m] > 0:
                edges.append({"from": int(k), "to": int(m), "label": "".format(adj_matrix[k][m])})

    print(nodes)
    nodes_json_str = json.dumps(nodes)
    edges_json_str = json.dumps(edges)
    final_js_str = "var nodeData = {0}; var edgeData = {1};".format(nodes_json_str, edges_json_str)
    return final_js_str


def _create_js_file(nodes_and_edges_str, file_name="html/nodes_and_edges.js"):
    with open(file_name, "w") as f:
        f.write(nodes_and_edges_str)


def _print_k(k, page_ranks, adj_matrix, image_id_array):
    nodes_and_edges_str = _create_nodes_and_edges_str(page_ranks, adj_matrix, image_id_array, k)
    _create_js_file(nodes_and_edges_str)
    print_message = "See visualization at: file:///{0}".format(os.path.abspath("html/phase3task3.html"))
    print(print_message)


def _page_rank(sim_mat):
    size = len(sim_mat)
    # Create markov matrix
    col_mat = csc_matrix(sim_mat, dtype=np.float)
    row_sums = np.array(col_mat.sum(1))[:, 0]
    rows, cols = col_mat.nonzero()
    col_mat.data /= row_sums[rows]
    sink = row_sums == 0
    sink_prob = sink / float(size)
    tele_prob = np.full(size, (1/size))

    # Go until convergence which is when little is gained
    ro, r = np.zeros(size), np.ones(size)
    it = 0
    while np.sum(np.abs(r-ro)) > min_gain and it < max_iters:
        it += 1
        ro = r.copy()
        for i in range(0, size):
            walk_prob = np.array(col_mat[:, i].todense())[:, 0]
            r[i] = ro.dot(walk_prob*walk + sink_prob*walk + tele_prob*(1-walk))
    # normalize the page rank
    return r/float(sum(r))
