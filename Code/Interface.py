from pathlib import Path
import pickle as pkl
import LoadData
import Task6Phase3
import Phase3Task5
import imagesearch
import Phase3Task1
import Phase3Task3
import Phase3Task2
import Phase3Task4
import os
invalidInputMessage = "Invalid number of input\n"

if __name__ == '__main__':
    # Check if pickled data exists to load..
    # Attempt to open a file and load (perfect case for try catch)
    try:
        with open('saveData.pickle', 'rb') as handle:
            userDict, imageDict, locationDict, topicsDict, visualDict, ids = pkl.load(handle)
    except Exception as e:
        userDict = LoadData.parseTextualDescriptors(LoadData.textualDescriptorFolder + "devset_textTermsPerUser.txt")
        imageDict = LoadData.parseTextualDescriptors(LoadData.textualDescriptorFolder + "devset_textTermsPerImage.txt")
        locationDict = LoadData.parseTextualDescriptors(LoadData.textualDescriptorFolder + "devset_textTermsPerPOI.txt")
        topicsDict = LoadData.parseTopics()
        visualDict = LoadData.normalizeData(LoadData.parseVisualDescriptors(topicsDict))
        ids = list(topicsDict.keys())
        for i in range(len(ids)):
            ids[i] = topicsDict[ids[i]]

    dicts = {}
    dicts['user'] = userDict
    dicts['location'] = locationDict
    dicts['image'] = imageDict
    dicts['visual'] = visualDict
    ids = list(topicsDict.keys())
    for i in range(len(ids)):
        ids[i] = topicsDict[ids[i]]
    termSpaceReduc = {}
    command = ""
    chosenDict = {}
    while command != "8":
        command = input("Commands (type number of command to execute):\n1. task 1\n2. task 2\n3. task 3\n4. task 4\n5. "
                        "task 5\n6. task 6\n7. quit\n> ")
        if command == "1":
            taskInput = input("\nEnter input for task 1 as follows: k\n>").split(' ')
            if len(taskInput) != 1:
                print(invalidInputMessage)
            else:
                print("Run task 1")
                Phase3Task1.main(taskInput[0], dicts)
        # insert code here
        elif command == "2":
            taskInput = input("\nEnter input for task 2 as follows (c-# of cluster, cluster_method-1=spectral 2=svd): c cluster_method\n>").split(' ')
            if len(taskInput) != 2:
                print(invalidInputMessage)
            else:
                # Test data load
                if os.path.isfile("imgAdjMatrixVisual.pickle"):  # changedv by Hardik
                    pickleFile = open("imgAdjMatrixVisual.pickle", 'rb')  # changedv by Hardik
                    df = pkl.load(pickleFile)
                    pickleFile.close()
                else:
                    df = Phase3Task1.task2help(15, visualDict)  # changedv by Hardik
                    pickleFile = open("imgAdjMatrixVisual.pickle", 'wb')
                    pkl.dump(df, pickleFile)
                    pickleFile.close()
                Phase3Task2.main(df, ids, visualDict, taskInput[0], taskInput[1])
                print("Done")
        elif command == "3":
            taskInput = input("\nEnter input for task 3 as follows: k\n>").split(' ')
            if len(taskInput) != 1:
                print(invalidInputMessage)
            else:
                print("Run task 3")
                try:
                    k = int(taskInput[0])
                    Phase3Task3.page_rank(k, imageDict)
                except ValueError as e:
                    print("K must be an integer. Error Message: \n{0}".format(e))
                except Exception as e:
                    print("An unknown error has occurred. "
                          "Please refer to the following error: \n{0}\n".format(e.__doc__))
        elif command == "4":
            taskInput = input("\nEnter input for task 4 as follows: imageid1 imageid2 imageid3 K\n>").split(' ')
            if len(taskInput) != 4:
                print(invalidInputMessage)
            else:
                Phase3Task4.main(taskInput[0], taskInput[1], taskInput[2], taskInput[3], imageDict)
        elif command == "5":
            vectors = []
            taskInput = input("\nEnter no. of layers, no. of hashes\n>").split(' ')
            if len(taskInput) != 2:
                print(invalidInputMessage)      

            hash_table, data_ids_to_index, rand_vectors, visual_vectors, bin_size, hash_keys = Phase3Task5.main(taskInput[0],taskInput[1],visualDict)

            image_id = input("\nEnter the image ID\n>")
            t = input("\nEnter the the value of t\n>")
            imagesearch.main(image_id,int(t),hash_table,data_ids_to_index,int(taskInput[0]), int(taskInput[1]),rand_vectors, visual_vectors, bin_size, hash_keys)


        elif command == "6":
            taskInput = input("\nEnter input for task 6 as follows: knnOrPPR k listOfImageLabelPairs\n>").split(' ')
            if len(taskInput) != 3:
                print(invalidInputMessage)
            else:
                print("Run task 6")
                Task6Phase3.task6Call(taskInput[0], taskInput[1], taskInput[2], imageDict)

        elif command == "7":
            print("Pickling will happen here once program ends.")
            # Pickling into single file as tuple object
            saveObject = (userDict, imageDict, locationDict, topicsDict, visualDict, ids)
            # Add logic to skip pickling if saveData exists
            save_file = Path('saveData.pickle')
            if not save_file.is_file():
                with open('saveData.pickle', 'wb') as handle:
                    pkl.dump(saveObject, handle)
            else:
                print('skipped pickling')
        else:
            print("Command not recognized.")