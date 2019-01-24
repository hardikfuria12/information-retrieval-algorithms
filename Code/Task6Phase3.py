import numpy as np
import os
import pickle
import Phase3Task1
import operator
from sklearn.decomposition import LatentDirichletAllocation
from scipy.sparse import csc_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy import spatial

# THE CONTROL BOARD
walk = 0.85
minGain = 0.04
maxIters = 30
# 175 seems like a sweet spot
ldaComp = 175
picsToShow = 100


def task6Call(functChoice, k, imageLabelList, imageDict):
    if functChoice != "knn" and functChoice != "ppr":
        print("Function choice does not exist.")
        return
    knownImages, knownLabels = parseimageLabelList(imageLabelList)
    unknownImages = getUnknownImages(knownImages, imageDict)
    termPos = termPosition(imageDict)
    allData = numpify(imageDict, termPos, "TF-IDF")
    allData = termSpaceLDA(allData)
    imagePos = imagePosition(imageDict)
    knownData = fetchImageData(knownImages, allData, imagePos)
    unknownData = fetchImageData(unknownImages, allData, imagePos)
    sortDir = False
    if functChoice == "knn":
        unknownLabels, scores = knn(int(k), knownData, knownLabels, unknownData)
    else:
        unknownLabels, scores = pprhelper(knownImages, knownData, knownLabels, unknownImages, unknownData, imageDict)
        sortDir = True
    scoreCopy = scores.copy()
    unknownLabels = sortUnknowns(unknownLabels, scoreCopy, sortDir)
    unknownImages = sortUnknowns(unknownImages, scores, sortDir)
    visualize(knownLabels, knownImages, unknownLabels, unknownImages, imageDict)
    print("View output in html page: \"task6Visualization.html\"")
    return


def sortUnknowns(labels, values, reverse):
    return [x for _,x in sorted(zip(values, labels), reverse=reverse)]


def parseimageLabelList(imageLabelList):
    images = []
    labels = []
    items = imageLabelList.split(";")
    for item in items:
        splitItem = item.split(":")
        images.append(splitItem[0])
        labels.append(splitItem[1])
    return images, labels


def getUnknownImages(knownImages, imageDict):
    allImages = list(imageDict.keys())
    for image in knownImages:
        if image in allImages:
            allImages.remove(image)
    allImages.sort()
    return allImages


def knn(k, knownData, knownLabels, unknownData):
    unknownLabels = []
    distanceScores = []
    # Go through each unknown image to label by comparing to each
    # known image
    for i in range(unknownData.shape[0]):
        distanceCalcs = []
        for j in range(knownData.shape[0]):
            distance = spatial.distance.cosine(knownData[j], unknownData[i])
            distanceCalcs.append((distance, knownLabels[j]))
        distanceCalcs.sort(key=operator.itemgetter(0))
        # Obtain topK
        topK = []
        for i in range(k):
            topK.append(distanceCalcs[i])
        # Count the classes. Also, keep track of distances to a class
        count = {}
        distance = {}
        for i in range(len(topK)):
            if topK[i][1] not in count:
                count[topK[i][1]] = 1
                distance[topK[i][1]] = topK[i][0]
            else:
                count[topK[i][1]] += 1
                distance[topK[i][1]] += topK[i][0]
        # Determine best related label based on the label
        # with most counts and ties are broken with the
        # smaller distance
        bestLabel = ""
        maxCount = 0
        minDistance = float('inf')
        for label in count:
            if (count[label] > maxCount) or (count[label] == maxCount and distance[label] < minDistance):
                maxCount = count[label]
                minDistance =  distance[label]
                bestLabel = label
        unknownLabels.append(bestLabel)
        distanceScores.append(minDistance)
    return unknownLabels, distanceScores


# Gets the tf-idf of specific images
def fetchImageData(ids, data, imagePos):
    i = 0
    positions = []
    for id in ids:
        positions.append(imagePos[id])
    return data[positions,:]


# Create a dictionary of images and relative positions
def imagePosition(dict):
    images = sorted(list(dict.keys()))
    imagePos = {}
    for i in range(len(images)):
        imagePos[images[i]] = i
    return imagePos


# Create a dictionary of terms and relative positions
def termPosition(dict):
    allTerms = []
    for entity in sorted(dict.keys()):
        for term in sorted(dict[entity]):
            allTerms.append(term)
    # Remove duplicates
    allTerms = set(allTerms)
    termPos = {}
    i = 0
    for term in sorted(allTerms):
        termPos[term] = i
        i += 1
    return termPos


# Convert dictionary of terms to list of terms
def dictToList(dict):
    termList = [None] * len(dict)
    for term, pos in dict.items():
        termList[pos] = term
    return termList


# Function to convert vector space into numpy array
def numpify(dict, termPos, model):
    matrix = lil_matrix((len(dict), len(termPos)))
    i = 0
    for entity in sorted(dict.keys()):
        for term in sorted(dict[entity].keys()):
            matrix[i, termPos.get(term)] = dict[entity][term][model]
        i += 1
    return csr_matrix(matrix)


def visualize(knownLabels, knownImages, classifiedLabels, classifiedImages, imageDict):
    uniqueLabels = set(classifiedLabels)
    page = open('task6Visualization.html','w')
    content = """<html><head></head><body>"""
    for label in uniqueLabels:
        content += "<table style=\"border: 2px solid black; margin:10px;\">" \
                   "<tr><th style=\"border: 2px solid black; margin:10px;\">Given for " + label + "</th>" \
                    "<th style=\"border: 2px solid black; margin:10px;\">Classified for " + label + "</th></tr>"
        content += "<tr><td style=\"border: 2px solid black; margin:10px;vertical-align: top;\">"
        for i in range(len(knownLabels)):
            if knownLabels[i] == label:
                content += "<img src=\"" + findDirectory(knownImages[i]) + "\"style=\"height:30%; width:30%;\"><br>"
        content += "</td><td style=\"border: 2px solid black; margin:10px;vertical-align: top;\">"
        count = 0
        for i in range(len(classifiedLabels)):
            if classifiedLabels[i] == label:
                content += "<img src=\"" + findDirectory(classifiedImages[i]) + "\"style=\"height:30%; width:30%;\"><br>"
                count += 1
            if count == picsToShow:
                break
        content += "</td></tr>"
        content += "</table>"
    content += """</body></html>"""
    page.write(content)
    page.close()
    return


def findDirectory(imageID):
    path = ""
    for subdir, dirs, files in os.walk('..\img'):
        for file in files:
            if imageID in file:
                path = os.path.join(subdir, file)
                break
    return path


def pprhelper(knownImages, knownData, knownLabels, unknownImages, unknownData, imageDict):
    # Run task 1 to get similarity matrix
    if os.path.isfile("task6input.pickle"): #changedv by Hardik
        pickleFile = open("task6input.pickle", 'rb')  #changedv by Hardik
        simMat = pickle.load(pickleFile)
        pickleFile.close()
    else:
        K, simMat = Phase3Task1.task6help(5, imageDict)  #changedv by Hardik
        pickleFile = open("task6input.pickle", 'wb')
        pickle.dump(simMat, pickleFile)
        pickleFile.close()
    unknownLabels = []
    prScores = []
    combinedImages = knownImages.copy()
    simMat.fillna(value=0,inplace=True)
    for uImage in unknownImages:
        combinedImages.append(uImage)
        indices = np.array(combinedImages)
        mat = simMat.loc[indices][indices].values
        for i in range(len(mat)):
            mat[i][i] = 0
        prVals = ppr(mat)
        prVals = prVals[:-1]
        maxIndex = prVals.argmax(axis=0)
        unknownLabels.append(knownLabels[maxIndex])
        prScores.append(prVals[maxIndex])
        del combinedImages[-1]
    return unknownLabels, prScores


def ppr(simMat):
    size = len(simMat)
    # Create markov matrix
    colMat = csc_matrix(simMat, dtype=np.float)
    rowSums = np.array(colMat.sum(1))[:, 0]
    rows, cols = colMat.nonzero()
    colMat.data /= rowSums[rows]
    sink = rowSums == 0
    sinkProb = sink / float(size)
    teleProb = np.zeros(size)
    teleProb[size-1] = 1
    # Go until convergence which is when little is gained
    ro, r = np.zeros(size), np.ones(size)
    it = 0
    while np.sum(np.abs(r-ro)) > minGain and it < maxIters:
        it += 1
        ro = r.copy()
        for i in range(0, size):
            walkProb = np.array(colMat[:,i].todense())[:,0]
            r[i] = ro.dot(walkProb*walk + sinkProb*walk + teleProb*(1-walk))
    # normalize the page rank
    return r/float(sum(r))


def termSpaceLDA(numpyArr):
    lda = LatentDirichletAllocation(n_components=ldaComp, random_state=0)
    newData = lda.fit_transform(numpyArr)
    components = lda.components_ / lda.components_.sum(axis=1)[:, np.newaxis]
    return newData
