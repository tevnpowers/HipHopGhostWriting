from jayz_willsmith_data import *
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import RFE, SelectKBest, chi2
from random import randint
import sys

#Features 'Line Count', 'Average Word Length', 'Average Line Length', 'Word Density', 'POS Density', 'Character 4-Gram Density', top words, POS rates
print('Finding most discriminatory features between Jay-Z and Will Smith...')
NUM_FEATURES = 10

all_song_points = jayz_points + willsmith_points
true_labels = [0]*len(jayz_points)+[1]*len(willsmith_points)

# since SelectKBest does not return the features in order of most effective to least, i loop through NUM_FEATURES time and find the 1st best feature, 2nd best feature, 3rd, etc. and append to feature_indices so they will be ordered.
feature_indices = []
for i in range(NUM_FEATURES):
	selector = SelectKBest(chi2, i+1)
	selector.fit(all_song_points, true_labels)

	new_indices = selector.get_support(indices=True)
	for index in new_indices:
		if index not in feature_indices:
			feature_indices.append(index)

print('Most discriminatory features are...')

for index in feature_indices:
	feature = feature_names[index]
	if feature.lower() in wsj_mapping.keys():
		key = wsj_mapping[feature.lower()]
		print(key + ': ' + wsj_to_description[key] + ' usage')
	elif feature in word_vocab:
		print('The word: ' + feature)
	else:
		print(feature)

print('Computing algorithm confidence...')

svc_predictions = []
# for each song in class 2, use SVM to determine if it best fits in class 1 or class 2 (should be class 2)
for i in range(len(all_song_points)):
	# target point information
	point = all_song_points[i]

	if i < len(jayz_points):
		# copy of class 1 points, with the target removed
		
		class1_copy = jayz_points[:]
		class1_copy.remove(point)

		# training points is the combination of all points minus the target
		data_points = class1_copy + willsmith_points
		
		# label class 1 songs as 0 and class 2 songs as 1 for training
		labels = ([0]*len(class1_copy)) + ([1]*len(willsmith_points))
	else:
		# copy of class 2 points, with the target removed
		class2_copy = willsmith_points[:]
		class2_copy.remove(point)

		# training points is the combination of all points minus the target
		data_points = jayz_points + class2_copy
	
		# label class 1 songs as 0 and class 2 songs as 1 for training
		labels = ([0]*len(jayz_points)) + ([1]*len(class2_copy))

	# Create multiple SVM variations
	h = 0.02
	C = 1.0

	svc = svm.SVC(kernel='linear', C=C).fit(data_points, labels)

	# Predict the label based off of the SVMs above
	svc_predicted_label = svc.predict([point])[0]
	svc_predictions.append(svc_predicted_label)

correct = 0.0
for i in range(len(svc_predictions)):
	if svc_predictions[i] == true_labels[i]:
		correct += 1.0

print('Confusion Matrix: ', confusion_matrix(true_labels, svc_predictions))
print('Classification Matrix: ', classification_report(true_labels, svc_predictions))

print('Percent Accuracy (confidence): ' + str(float(correct*100/len(svc_predictions))) + '%')
print('Program complete...')