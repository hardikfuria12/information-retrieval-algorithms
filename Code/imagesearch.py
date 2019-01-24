import Phase3Task5
import numpy as np
from scipy.spatial import distance
import heapq
import os

def getNeighboursFromBucket(binned_values,hash_table,no_layers,no_hashes):
	neighbours = set()
	for i in range(no_layers):
		hash_key_list = binned_values[0,i*no_hashes:(i+1)*no_hashes].tolist()
		hash_key_str = ','.join(str(c) for c in hash_key_list)
		try:
			neighbours = neighbours.union(set(hash_table[i][hash_key_str]))		
		#	print(neighbours)		
		except:	
			#print("No such bucket")		
			pop=1
	return neighbours

def getNeighbours(binned_values,hash_table,no_layers,no_hashes,layer_no):
	neighbours = set()	
	hash_key_list = binned_values[0,layer_no*no_hashes:(layer_no+1)*no_hashes].tolist()
	hash_key_str = ','.join(str(c) for c in hash_key_list)
	try:
		neighbours = neighbours.union(set(hash_table[layer_no][hash_key_str]))		
	#	print(neighbours)		
	except:	
		#print("No such bucket")		
		pop=1
	return neighbours


def getKNN(visual_vectors, t_neighbours_set, t, query_index, index_to_data_ids):
	distance_list=[]
	sorted_distance_IDs=[]
	for index in t_neighbours_set:		
		key = str(index_to_data_ids[index])
		dist = distance.euclidean(visual_vectors[index], visual_vectors[query_index])
		distance_list.append([key,dist])

	sorted_distances=sorted(distance_list,key =lambda k: k[1])

	for item in sorted_distances[:t]:
		sorted_distance_IDs.append(item[0])

	return sorted_distance_IDs

def perturbationScores(projected_vector,no_layers,no_hashes,bin_size):
	not_floored_binned_values = np.divide(projected_vector,bin_size)
	
	binned_values = np.floor(not_floored_binned_values)

	one_minus =projected_vector - (binned_values * bin_size)
	one_plus = bin_size - one_minus
	
	perturbation_scores={}

	for i in range(no_layers):
		perturbation_scores[i] = []
		for j in range(i*no_hashes,(i+1)*no_hashes):
			## element = list(score,index in binned values, 1/-1, layerno)
			element = [one_minus[0,j], j, -1, i]
			perturbation_scores[i].append(element)
			element = [one_plus[0,j], j, 1, i]
			perturbation_scores[i].append(element)

	for i in range(no_layers):
		perturbation_scores[i] = sorted(perturbation_scores[i],key =lambda k: k[0])	
	return perturbation_scores

def shiftOperation(minimum_perturb_indices,perturbation_scores):
	indices_list = minimum_perturb_indices[1].copy()
	indices_list[-1] = indices_list[-1] + 1
	new_score = 0
	layer_no = minimum_perturb_indices[2]
	for index in indices_list:
		new_score = new_score + perturbation_scores[layer_no][index][0]

	new_minimum_perturb_indices = (new_score, indices_list, layer_no)
	return new_minimum_perturb_indices

def expandOperation(minimum_perturb_indices,perturbation_scores):
	indices_list = minimum_perturb_indices[1].copy()	
	indices_list.append(indices_list[-1]+1)	
	new_score = 0
	layer_no = minimum_perturb_indices[2]
	for index in indices_list:
		new_score = new_score + perturbation_scores[layer_no][index][0]		

	new_minimum_perturb_indices = (new_score, indices_list, layer_no)
	return new_minimum_perturb_indices

def checkValidity(minimum_perturb_indices,bin_size,perturbation_scores,no_hashes):
	indices_list = minimum_perturb_indices[1]

	exist_set = set()

	for index in indices_list:
		if index>=2*no_hashes:
			#print("Invalid")
			return False
		layer = minimum_perturb_indices[2]
		if perturbation_scores[layer][index][1] in exist_set:
			#print("Invalid")
			return False
		exist_set.add(index)

	return True

min_heap = []
heap_set = set()


def generatePerturbationVectors(no_layers,no_hashes,perturbation_scores,bin_size):
	while (True):
		minimum_perturb_indices = heapq.heappop(min_heap)		
		try:			
			shifted_minimum_perturb_indices = shiftOperation(minimum_perturb_indices,perturbation_scores)
			key = str(shifted_minimum_perturb_indices[0])+"+"+str(shifted_minimum_perturb_indices[1])+"+"+str(shifted_minimum_perturb_indices[2])
			if key not in heap_set:
				heapq.heappush(min_heap,shifted_minimum_perturb_indices)
				heap_set.add(key)
		except:
			#print("shift fail")
			loka=1
		try:
			expanded_minimum_perturb_indices = expandOperation(minimum_perturb_indices,perturbation_scores)
			key = str(expanded_minimum_perturb_indices[0])+"+"+str(expanded_minimum_perturb_indices[1])+"+"+str(expanded_minimum_perturb_indices[2])
			if key not in heap_set:
				heapq.heappush(min_heap,expanded_minimum_perturb_indices)
				heap_set.add(key)
		except:
			#print("expand fail")
			loka=1

		heapq.heapify(min_heap)
		if checkValidity(minimum_perturb_indices,bin_size,perturbation_scores,no_hashes) == True:
			break;

	return minimum_perturb_indices

def applyPerturbation(perturbation_indices,binned_values,perturbation_scores,no_hashes):
	indices = perturbation_indices[1]
	layer = perturbation_indices[2]
	for index in indices:
		index_in_binned_values = perturbation_scores[layer][index][1]
		binned_values[0,index_in_binned_values] = binned_values[0,index_in_binned_values] + perturbation_scores[layer][index][2]
	return binned_values

def findSimilarHash(hash_table,binned_values, hash_keys,no_layers, no_hashes):
	similar_hashes = []	
	for i in range(no_layers):
		for hash_key in hash_keys[i]:
			#print("No of elements in hash table"+str(i))
			#print(len(hash_keys[i]))
			if np.array_equal(np.array(hash_key), binned_values[0,i*no_hashes:(i+1)*no_hashes]) == False:
				sim = distance.euclidean(np.array(hash_key), binned_values[0,i*no_hashes:(i+1)*no_hashes])
				##### [similarity, key as a list, layer_no]				
				similar_hashes.append([sim, np.array(hash_key), i])
	similar_hashes = sorted(similar_hashes,key =lambda k: k[0])	
	return similar_hashes

def findDirectory(imageID):
	path = ""
	for subdir, dirs, files in os.walk('../img'):
		for file in files:
			if imageID in file:
				path = os.path.join(subdir, file)
				break
	return path

def visualize(query_ID, sorted_distance_IDs):	
	page = open('task5Visualization.html','w')
	content = """<html><head></head><body>"""
	content += "<table style=\"border: 2px solid black; margin:10px;\">" \
			   "<tr><th style=\"border: 2px solid black; margin:10px;\">Query" + "</th>" \
			   "<th style=\"border: 2px solid black; margin:10px;\">Similar Images" + "</th></tr>"
	content += "<tr><td style=\"border: 2px solid black; margin:10px;vertical-align: top;\">"
	content += "<figure>" \
				"<img src=\"" + findDirectory(str(query_ID)+".jpg") + "\"style=\"height:30%; width:30%;\"><br>" \
				"<figcaption>" + str(query_ID) + "</figcaption>" \
				"</figure>"

	content += "</td><td style=\"border: 2px solid black; margin:10px;vertical-align: top;\">"
	for i in range(len(sorted_distance_IDs)):        
		content += "<figure>" \
				"<img src=\"" + findDirectory(str(sorted_distance_IDs[i])+".jpg") + "\"style=\"height:30%; width:30%;\"><br>" \
				"<figcaption>" + str(sorted_distance_IDs[i]) + "</figcaption>" \
				"</figure>"		
	content += "</td></tr>"
	content += "</table>"
	content += """</body></html>"""
	page.write(content)
	page.close()
	return

def main(image_id,t,hash_table,data_ids_to_index,no_layers,no_hashes,rand_vectors,visual_vectors,bin_size, hash_keys):
	t_neighbours_set = set()
	index = data_ids_to_index[image_id]
	vector = np.reshape(visual_vectors[index][:],(1,-1))
	
	total_images = 0

	projected_vector = Phase3Task5.projectToVectors(rand_vectors,vector)
	perturbation_scores = perturbationScores(projected_vector,no_layers,no_hashes,bin_size)
	binned_values = Phase3Task5.putInBins(projected_vector,bin_size)

	for i in range(no_layers):
		#### heap contains the following tuple (score,[indices],layer_no)
		min_heap.append((perturbation_scores[i][0][0],[0],i))
		heap_set.add(str(perturbation_scores[i][0][0])+"+"+str([0])+"+"+str(i))
	heapq.heapify(min_heap)

	flag=0
	while(len(t_neighbours_set) < int(t)):		
		if flag == 0:		
			neighbours_set = getNeighboursFromBucket(binned_values,hash_table,no_layers,no_hashes)
			flag=1
		elif flag == 1:
			neighbours_set = getNeighbours(binned_values,hash_table,no_layers,no_hashes,layer_no)
		print("Found "+str(len(neighbours_set))+" neighbours in nearby bucket")
		total_images = len(neighbours_set) + total_images
		t_neighbours_set = t_neighbours_set.union(neighbours_set)
		perturbation_indices = generatePerturbationVectors(no_layers,no_hashes,perturbation_scores,bin_size)
		binned_values = applyPerturbation(perturbation_indices,binned_values,perturbation_scores,no_hashes)
		layer_no = perturbation_indices[2]


	# ####### Second Method
	# neighbours_set = getNeighboursFromBucket(binned_values,hash_table,no_layers,no_hashes)
	# t_neighbours_set = neighbours_set
	# total_images = len(t_neighbours_set)
	# if (len(neighbours_set) < int(t)):
	# 	print("Finding similar buckets")
	# 	similar_hashes = findSimilarHash(hash_table, binned_values, hash_keys, no_layers, no_hashes)
	# 	i=1
	# 	while(len(t_neighbours_set) < int(t)):
	# 		print("New neighbours size"+str(len(t_neighbours_set)))
	# 		most_similar_hash = similar_hashes[i][1]
	# 		most_similar_hash_layer = similar_hashes[i][2]

	# 		new_binned_values = binned_values
	# 		new_binned_values[0,most_similar_hash_layer*no_hashes:(most_similar_hash_layer+1)*no_hashes] = most_similar_hash
	# 		neighbours_set = getNeighboursFromBucket(new_binned_values, hash_table, no_layers, no_hashes)
	# 		total_images = len(neighbours_set) + total_images
	# 		t_neighbours_set = t_neighbours_set.union(neighbours_set)
	# 		i = i+1

	index_to_data_ids = {v: k for k, v in data_ids_to_index.items()}
	sorted_distance_IDs = getKNN(visual_vectors, t_neighbours_set, t, index, index_to_data_ids)

	print("Unique images searched through: "+str(len(t_neighbours_set)))
	print("Total images searched through: "+str(total_images))
	print("Done")
	print("Writing to html file")
	visualize(image_id, sorted_distance_IDs)