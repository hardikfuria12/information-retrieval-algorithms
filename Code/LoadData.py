import xml.etree.ElementTree as ET
import csv
import StringConstants

# CHANGE THIS FILE PATH TO THE DIRECTORY CONTAINING ALL THE FILES

projectFolder = "C:\\Users\\amd30\\Downloads\\cse515\\"
textualDescriptorFolder = projectFolder + "desctxt/"
visualDescriptorFolder = projectFolder + "descvis/img/"
topics = projectFolder + "devset_topics.xml"


# Since the visual features will go through euclidean similarity,
# it is good to normalize the data. This is what this function achieves.
def normalizeData(visualDict):
    # Need to pull a random row out to get number of features of model
    locationKey = next(iter(visualDict))
    imageKey = next(iter(visualDict[locationKey][StringConstants.visualKeys[0]]))
    for model in StringConstants.visualKeys:
        modelLength = len(visualDict[locationKey][model][imageKey])
        minVal = [float('inf')] * modelLength
        maxVal = [float('-inf')] * modelLength
        for location in visualDict:
            for image, data in visualDict[location][model].items():
                for i in range(modelLength):
                    minVal[i] = min(minVal[i], float(data[i]))
                    maxVal[i] = max(maxVal[i], float(data[i]))
        for location in visualDict:
            for image in visualDict[location][model]:
                for i in range(modelLength):
                    visualDict[location][model][image][i] = float((float(visualDict[location][model][image][i]) - minVal[i])/(maxVal[i] - minVal[i]))
    return visualDict


# This function goes through the textual descriptor files and puts them into a data structure
def parseTextualDescriptors(filePath):
    with open(filePath, 'r', encoding='utf-8') as textDescriptor:
        data = textDescriptor.read()
        entities = data.split("\n")
        entities = entities[:-1]

        keyDict = {};
        for entity in entities:
            termDict = {}
            endOfKey = entity.find(" \"")
            key = entity[:endOfKey]
            restOfInfo = entity[endOfKey+1:-1].split(" ")
            for i in range(0, len(restOfInfo), 4):
                term = restOfInfo[i].replace("\"", "")
                freqDict = {}
                for j in range(len(StringConstants.textualKeys)):
                    freqDict[StringConstants.textualKeys[j]] = float(restOfInfo[i+j+1])
                termDict[term] = freqDict
            keyDict[key] = termDict

    return keyDict


# This function goes through the visual descriptor files and puts them into a data structure
def parseVisualDescriptors(topicsDict):
    locationDict = {}
    for i in range(1, len(topicsDict)+1):
        modelDict = {}
        for j in range(len(StringConstants.visualKeys)):
            imageDict = {}
            with open(visualDescriptorFolder + topicsDict[str(i)] + " " + StringConstants.visualKeys[j] + ".csv", 'r') as sheet:
                reader = csv.reader(sheet)
                rows = list(reader)
                for row in rows:
                    imageID = row.pop(0)
                    imageDict[imageID] = row
            modelDict[StringConstants.visualKeys[j]] = imageDict
        locationDict[topicsDict[str(i)]] = modelDict
    return locationDict

# This will go through each of the location ids to get the actual name of the location.
def parseTopics():
    topicsDict = {}
    topicsRoot = ET.parse(topics).getroot()
    for topic in topicsRoot:
        topicsDict[topic.find('number').text] = topic.find('title').text
    return topicsDict
