import numpy as np
import os
import pickle
import Phase3Task1
import sys
from scipy.sparse import csc_matrix

walk = 0.85
min_gain = 0.04
max_iters = 30


def get_image_image_similarity_matrix(image_dict):
    pickle_file_name = "imgAdjMatrixTextual.pickle"
    if os.path.isfile(pickle_file_name):
        pickleFile = open(pickle_file_name, 'rb')
        sim_mat = pickle.load(pickleFile)
        pickleFile.close()
    else:
        k, sim_mat = Phase3Task1.main(len(image_dict) - 1, image_dict)
        pickle_file = open(pickle_file_name, 'wb')
        pickle.dump(sim_mat, pickle_file)
        pickle_file.close

    sim_mat.fillna(value=0, inplace=True)
    return sim_mat


def ppr(sim_mat, index1, index2, index3):
    image_similarity_matrix = sim_mat.copy()
    image_similarity_matrix = image_similarity_matrix.values
    for i in range(0, len(image_similarity_matrix)):
        image_similarity_matrix[i][i] = 0

    size = len(image_similarity_matrix)
    # Create markov matrix
    col_mat = csc_matrix(image_similarity_matrix, dtype=np.float)
    row_sums = np.array(col_mat.sum(1))[:, 0]
    rows, cols = col_mat.nonzero()
    col_mat.data /= row_sums[rows]
    sink = row_sums == 0
    sink_prob = sink / float(size)
    tele_prob = np.full(size, 0.0)
    tele_prob[index1] = float(1 / 3)
    tele_prob[index2] = float(1 / 3)
    tele_prob[index3] = float(1 / 3)

    ro, r = np.zeros(size), np.ones(size)
    it = 0
    while np.sum(np.abs(r - ro)) > min_gain and it < max_iters:
        it += 1
        ro = r.copy()
        for i in range(0, size):
            walk_prob = np.array(col_mat[:, i].todense())[:, 0]
            r[i] = ro.dot(walk_prob * walk + sink_prob * walk + tele_prob * (1 - walk))
    # normalize the page rank
    return r / float(sum(r))


def findDirectory(imageID):
    path = ""
    for subdir, dirs, files in os.walk('../img'):
        for file in files:
            if imageID + '.jpg' in file:
                path = os.path.join(subdir, file)
                break
    return path


def visualize(image_ids):
    page = open('task4Visualization.html', 'w')
    content = """<html><head></head><body>"""
    for imageid in image_ids:
        content += "<table style=\"border: 2px solid black; margin:10px;\">" \
            #   "<tr><th style=\"border: 2px solid black; margin:10px;\">Given for  </th>"\
        # "<th style=\"border: 2px solid black; margin:10px;\">Classified for </th></tr>"
        content += "<tr><td style=\"border: 2px solid black; margin:10px;vertical-align: top;\">"
        content += "<img src=\"" + findDirectory(str(imageid)) + "\"style=\"height:100%; width:100%;\"><br>"
        content += "</td><td style=\"border: 2px solid black; margin:10px;vertical-align: top;\">"
        content += "</td></tr>"
        content += "</table>"
    # val = findDirectory(str(imageid))
    # print(val, "this is the path")
    content += """</body></html>"""
    page.write(content)
    page.close()
    return


def main(imageid1, imageid2, imageid3, k, image_dict):
    imgids = []
    retval = get_image_image_similarity_matrix(image_dict)
    list_of_indices = list(map(int, retval.index.tolist()))
    # print(len(list_of_indices))
    # print(type(list_of_indices[0]))
    pagerankvals = ppr(retval, list_of_indices.index(float(imageid1)), list_of_indices.index(float(imageid2)),
                       list_of_indices.index(float(imageid3)))
    top_k_idx = pagerankvals.argsort()[-int(k):][::-1]
    for i in range(len(top_k_idx)):
        imgids.append(list_of_indices[top_k_idx[i]])
    visualize(imgids)
# print(path, "this is the path")
# for idx in top_k_idx:
# 	print(idx, pagerankvals[idx])
