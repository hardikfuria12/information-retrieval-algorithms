import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import operator
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from pathlib import Path
import StringConstants

from scipy.spatial.distance import pdist, squareform

# Try different conversions
def convert_tfidf_vals(elem):
    if isinstance(elem, dict):
        return elem['TF-IDF']
    else:
        return


def cleanImageTextMatrix(imageDict):
    df = pd.DataFrame.from_dict(imageDict, orient='columns')
    # Select out tfidf values to fill df
    tfidf_df = df.applymap(convert_tfidf_vals)
    low_count = []
    for col in tfidf_df.T.columns:
        if len(tfidf_df.T[col].value_counts()) <= 1:
            low_count.append(col)
    trim_idf = tfidf_df.drop(low_count)
    trim_idf.fillna(0, inplace=True)
    image_terms = csr_matrix(trim_idf.T.values)
    image_list = list(trim_idf.T.index)

    return image_terms, image_list


def im_im_similarity_matrix(image_terms):
    # No decompositions applied
    K = cosine_similarity(image_terms)
    return K

def imageAdjacenyMatrixT(K,image_list, k):

    imgSimMatrix=pd.DataFrame(K,index=image_list,columns=image_list,dtype=float)
    rowWise=imgSimMatrix.to_dict('records')
    imgAdjMatrix=imgSimMatrix.copy()
    imgAdjMatrix[:]=np.nan
    # imageList=imgSimMatrix.index.tolist()
    imgAdjList=pd.DataFrame(index=image_list,columns=['SimImages'])
    imgAdjDict={}
    i=0
    for row in rowWise:
        sorted_z = sorted(row.items(), key=operator.itemgetter(1),reverse=True)
        # sorted_z = sorted_z[:k+1]
        simimglist=[]
        length_sim=0
        rowid = str(image_list[i])
        # imgAdjMatrix.loc[rowid][rowid]=1
        for zzz in sorted_z:
            imgid=str(zzz[0])
            if imgid!=rowid and (length_sim<k):
                imgAdjMatrix.loc[rowid][imgid] = float(zzz[1])
                simimglist.append(imgid)
                length_sim+=1
        if len(simimglist)!=k:
            print("Some mistake")
        # imgAdjMatrix.loc[rowid][rowid] = 1
        imgAdjList.loc[rowid]['SimImages']=simimglist
        imgAdjDict[rowid]=simimglist
        i = i + 1
    return imgAdjMatrix,imgAdjList,imgAdjDict


def imageAdjacenyMatrixV(imgSimMatrix,k):

    rowWise=imgSimMatrix.to_dict('records')
    imgAdjMatrix=imgSimMatrix.copy()
    imgAdjMatrix2=imgSimMatrix.copy()
    imgAdjMatrix[:]=np.nan
    imgAdjMatrix2[:]=0
    imageList=imgSimMatrix.index.tolist()
    imgAdjList=pd.DataFrame(index=imageList,columns=['SimImages'])
    imgAdjDict={}
    i=0
    for row in rowWise:
        sorted_z = sorted(row.items(), key=operator.itemgetter(1))
        simimglist=[]
        length_sim = 0
        rowid = str(imageList[i])
        for zzz in sorted_z:
            imgid=str(zzz[0])
            if imgid!=rowid and (length_sim<k):
                imgAdjMatrix.loc[rowid,imgid] = float(zzz[1])
                imgAdjMatrix2.loc[rowid,imgid] = float(zzz[1])
                imgAdjMatrix2.loc[imgid,rowid] = float(zzz[1])
                simimglist.append(imgid)
                length_sim+=1
        if len(simimglist)!=k:
            print('Somethings no right')
        #imgAdjList.loc[rowid,'SimImages']=simimglist
        imgAdjDict[rowid]=simimglist
        i = i + 1
    return imgAdjMatrix,imgAdjList,imgAdjDict,imgAdjMatrix2




# FOR A GIVEN MODEL CREAETES A DATAFRAME WITH IMAGEIDS AS INDEXES AND COLUMNS ARE THE VIDSUAL DESCRIPTOR FEATURES.
def createAllModelMatrix(visualDict):
    objectFeatureMatrix = pd.DataFrame()
    for model in StringConstants.visualKeys:
        modelBasedDf = pd.DataFrame()
        for locId in visualDict:
            imageValueDict = visualDict[locId][model]
            interDf = pd.DataFrame.from_dict(imageValueDict, orient='index')
            modelBasedDf = modelBasedDf.append(interDf)
        objectFeatureMatrix = pd.concat([objectFeatureMatrix, modelBasedDf], axis=1)
    return objectFeatureMatrix


def computeImageSimilarityMatrix(objectFeatureMatrix):
    allImageList = objectFeatureMatrix.index.tolist()


    dist = pdist(objectFeatureMatrix, 'euclidean')
    locSimilarityMatrix = pd.DataFrame(squareform(dist), index=allImageList, columns=allImageList)
    return locSimilarityMatrix


def savepickle(var,filename):
    filename=filename+'.pickle'
    save_file = Path(filename)
    if not save_file.is_file():
        with open(filename, 'wb') as handle:
            pkl.dump(var, handle)
    print(filename," was saved")


def task6help(k,imageDict):
    # print('here')

    image_terms, image_list = cleanImageTextMatrix(imageDict)
    K = im_im_similarity_matrix(image_terms)
    task6input=pd.DataFrame(K,index=image_list,columns=image_list)
    return K,task6input


def task34help(k,imageDict):
    k=int(k)
    image_terms, image_list = cleanImageTextMatrix(imageDict)
    K = im_im_similarity_matrix(image_terms)
    task6input = pd.DataFrame(K, index=image_list, columns=image_list)
    imgAdjMatrixTextual, imgAdjList, imgAdjDict = imageAdjacenyMatrixT(K, image_list, k)
    # print(task6input)
    return K,imgAdjMatrixTextual


def task2help(k,visualDict):
    k=int(k)
    objectFeatureMatrix = createAllModelMatrix(visualDict)
    imgSimMatrix = computeImageSimilarityMatrix(objectFeatureMatrix)
    imgAdjMatrixVisual, imgAdjList, imgAdjDict, imgAdjMatrixVisual2 = imageAdjacenyMatrixV(imgSimMatrix, k)
    # print("Directed and Weighted Adjacency Matrix based on Visual Descriptors")
    # print(imgAdjMatrixVisual)
    return imgAdjMatrixVisual

def main(k, dicts):
    k=int(k)
    imageDict=dicts['image']
    visualDict=dicts['visual']

    #for visual model
    objectFeatureMatrix = createAllModelMatrix(visualDict)
    imgSimMatrix = computeImageSimilarityMatrix(objectFeatureMatrix)
    imgAdjMatrixVisual, imgAdjList, imgAdjDict,imgAdjMatrixVisual2 = imageAdjacenyMatrixV(imgSimMatrix, k)
    print("Directed and Weighted Adjacency Matrix based on Visual Descriptors")
    print(imgAdjMatrixVisual)

    savepickle(imgAdjMatrixVisual,'imgAdjMatrixVisual')



    # for image model
    image_terms, image_list = cleanImageTextMatrix(imageDict)
    K = im_im_similarity_matrix(image_terms)
    task6input=pd.DataFrame(K,index=image_list,columns=image_list)
    imgAdjMatrixTextual, imgAdjList, imgAdjDict = imageAdjacenyMatrixT(K, image_list, k)
    # print(task6input)
    print("Directed and Weighted Adjacency Matrix based on Textual Descriptors")
    print(imgAdjMatrixTextual)
    savepickle(task6input,'task6input')
    savepickle(imgAdjMatrixTextual,'imgAdjMatrixTextual')










    # Takes in the imageDict and k value
    # Builds the K pairwise image similarity matrix

    # =======================================
    #
    # save_file = Path('imgAdjMatrix.pickle')
    # print(imgAdjMatrix)
    # print(imgAdjDict)
    # if not save_file.is_file():
    #     with open('imgAdjMatrix.pickle', 'wb') as handle:
    #         pkl.dump(imgAdjMatrix, handle)
    #
    # save_file = Path('imgAdjDict.pickle')
    # if not save_file.is_file():
    #     with open('imgAdjDict.pickle', 'wb') as handle:
    #         pkl.dump(imgAdjDict, handle)
    # return K,imgAdjMatrix
    # ==========================

# ================================================
# def createGraph(image_list,imgAdjMatrix):
#     imgAdjMatrix=imgAdjMatrix.values
#     g = nx.Graph()
#     for i in range(imgAdjMatrix.shape[0]):
#         g.add_node(i)
#
#     for i in range(imgAdjMatrix.shape[0]):
#         for j in range(imgAdjMatrix.shape[0]):
#             if imgAdjMatrix[i][j] == 1:
#                 g.add_edge(i, j)
#
#     print(nx.info(g))
#     nx.draw(g, with_labels=True)
#     plt.show()
# =============================================

