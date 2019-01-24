import numpy as np
import sklearn
from scipy.spatial import distance

hash_table = {}

## Create a numpy array out of the visualDict dictionary for all images
def createArray(visualDict):
	visualKeys = ["HOG","LBP3x3","CSD"]
	data_array = []
	data_ids_to_index={}
	vector_complete_status=[]
	index=0

	vector_length = 289

	for location in visualDict:	
		for visualModel in visualKeys:	
			for image_ID in visualDict[location][visualModel]:				
					try:
						if len(data_array[data_ids_to_index[image_ID]] + visualDict[location][visualModel][image_ID]) <= vector_length:
							data_array[data_ids_to_index[image_ID]] = data_array[data_ids_to_index[image_ID]] + visualDict[location][visualModel][image_ID]
					except:
						data_ids_to_index[image_ID]=index
						data_array.append(visualDict[location][visualModel][image_ID])
						index=index+1
	
	data_array = np.array(data_array)
	#print(data_array)
	#print(data_array.shape)
	return data_array,data_ids_to_index

def createRandomVectors(n,m):
	rand_vectors = np.random.randn(n,m)
	#Need to normalize the negative values !!!!	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	row_sum = rand_vectors[:,:m-1].sum(axis=1)
	normalized_rand_vectors = rand_vectors[:,:m-1] / row_sum[:, np.newaxis]

	rand_vectors[:,m-1] *= 50
	#print(rand_vectors)
	return rand_vectors

def projectToVectors(rand_vectors,vectors):
	new_vectors = np.full((vectors.shape[0], vectors.shape[1]+1), 1)
	new_vectors = new_vectors.astype(float)
	new_vectors[:,:vectors.shape[1]] = vectors
	#print(new_vectors)
	return np.dot(new_vectors,rand_vectors.T)

def putInBins(projected_vectors,bin_size):
	binned_values = np.divide(projected_vectors,bin_size)
	binned_values = np.floor(binned_values)
	return binned_values

def createHashTable(vectors,binned_values,no_layers,no_hashes):
	hash_keys = []
	for i in range(no_layers):
		hash_table[i]={}
		hash_keys.append([])

	for j in range(binned_values.shape[0]):
		for i in range(no_layers):
			row_as_list = binned_values[j,i*no_hashes:(i+1)*no_hashes].tolist()
			row_as_str = ','.join(str(c) for c in row_as_list)			
			try:
				## Appending indexes of the vectors instead of the vectors themselves into the hash table
				hash_table[i][row_as_str].append(j)
			except:
				hash_table[i][row_as_str] = []
				hash_table[i][row_as_str].append(j)
				hash_keys[i].append(row_as_list)
	return hash_keys
				

def main(no_layers,no_hashes,visualDict):
	visual_vectors, data_ids_to_index = createArray(visualDict)
	vectors = visual_vectors
	no_hashes=int(no_hashes)
	no_layers=int(no_layers)
	rand_vectors = createRandomVectors(no_hashes*no_layers, vectors.shape[1]+1)
	projected_vectors = projectToVectors(rand_vectors,vectors)
	bin_size = 5
	binned_values = putInBins(projected_vectors,bin_size)
	print(binned_values.shape)
	hash_keys = createHashTable(vectors,binned_values,no_layers,no_hashes)
	#print(hash_table)
	#print(len(hash_table))
	return hash_table, data_ids_to_index, rand_vectors, visual_vectors, bin_size, hash_keys
