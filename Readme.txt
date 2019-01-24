1. Folder Code contains the code
2. Please make sure you have the 'img' data set folder in the root directory for the visualziation code to work properly for the above mentioned Tasks.
"Root Directory"
-"Code"
-"img"
-"DatasetFolder"
5. Dataset Folder must include the folders "desctxt" and "descvis" availabel in this link - http://skuld.cs.umass.edu/traces/mmsys/2015/paper-5/devset/
6.Edit the LoadData.py script to correctly point to the above two mention folders.For this set the variable "projectFolder" = "RootDirectory/DatasetFolder"
7. Run Interface.py
8. Execute the different tasks as required.
9. Outputs will be generated for  task 2,3,4,5,6 and their results will be generated inside the "Code" Folder


TASK DESCRIPTIONS
1.This task is required for functioning of other tasks. Given a value k, this task, creates an image-image similarity graph, such that from each image, there are k outgoing edges to k most similar/related images to it.
2.Using task 1 generated graph, this task identifies c clusters in the given graph using  Spectral Clustering and Svd-based Angular Clustering.
3.Identifies and creates visualzization for K most dominant images in the given graph from task 1 b applying Page Rank Algorithms. 
4.This task is like the Task 3 but instead of the global dominant images this task finds the K most dominant images to a given Image. In this sense this task is implemeneted as Personalized Page Rank
5.In this task Local Sensitive Hashing is implemented, and based on the LSH created K most similar images are found.
6.This task is Page Rank based Classification.


   
 
 