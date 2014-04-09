# algorithm source: http://www.clef-initiative.eu/documents/71612/d212239b-1138-4a5b-82d9-a206313f16ae

import numpy as np
from math import sqrt

# find the k nearest neighbors to the primary point
def find_knn(k, primary, neighbors):
	distances = []
	for i in range(len(neighbors)):
		distance = euclidean_distance(primary, neighbors[i])
		distances.append(distance)
	distances = np.array(distances)
	sorted_distances = np.argsort(distances)
	
	return sorted_distances[:k]

# compute the euclidean distance between two points
def euclidean_distance(point1, point2):
	sum = 0.0
	for i in range(len(point1)):
		sum += (point1[i]-point2[i])**2
	return sqrt(sum)
	
# test code
if __name__ == '__main__':
	a = [0, 1, 2, 3]
	b = [1, 2, 3, 4]
	c = [100., 1, 2, 3]
	d = [100., 2, 3, 4]
	e = [0, 1, 2, 3]

	neighbors = [b, c, d, e]
	print(find_knn(4, a, neighbors))