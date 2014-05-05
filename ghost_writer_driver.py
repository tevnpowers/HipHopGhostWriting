from file_parser import *
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from random import randint

# Command Line: python ghost_writer_driver.py Dr_Dre Dr_Dre_GhostWritten

# check for correct parameters
if len(sys.argv) < 3:
	print('You must provide the following parameters: 1) Name of directory containing texts written by author one 2) Name of directory containing ghost written texts attributed to author one.')
	sys.exit()

# determine if the first parameter is a directory, if not then notify the user and exit the program
if os.path.isdir(sys.argv[1]):
	class1_song_names, class1_songs = get_all_songs(sys.argv[1])
else:
	print('Unknown parameter ' + sys.argv[1] + '. Pleaes provide a file name or directory.')
	sys.exit()

# determine if the second parameter is a directory, if not then notify the user and exit the program
if os.path.isdir(sys.argv[2]):
	class2_song_names, class2_songs = get_all_songs(sys.argv[2])
else:
	print('Unknown parameter ' + sys.argv[2] + '. Pleaes provide a file name or directory.')
	sys.exit()

# which features to extract from each song
# word rates, character grams, word grams, avg word length, and avg line length
feature_list = [get_words, get_ngrams, get_word_grams, get_avg_word_length, get_avg_line_length]

# what n should be (only necessary for character gram and word gram extraction
n_list = [None, 3, 2, None, None]

# given a list of lyrics and list of song names, create high-dimensional data points to respresent each point based on the features defined in feature_list
def create_classification_points(songs, song_names):
	# list to contain the data point of each song
	song_points = []
	
	# each data point is represented as a list (the list will dynamically grow with rates from each feature)
	for song in songs:
		song_points.append([])

	point_len = 0
	# loop through each feature function in the given list, adding the feature values to the data point for each song
	for i in range(len(feature_list)):
		feature_func = feature_list[i]

		# build the representative points for each feature (the dtm holds the feature data for each song)
		if feature_func in [get_song_length, get_avg_word_length, get_line_count, get_avg_line_length]:
			dtm, documents = build_single_feature(song_names, songs, feature_func)
		else:
			dtm, vocab, documents = build_feature_vocab(n_list[i], song_names, songs, feature_func)

		'''
		if i <= 2:
			point_len += len(dtm[0])
			
		if i == 2:
		avg_point_len = point_len/3.0
		'''
			
		# add the values for this feature set to the high dimensional point for each song
		for j in range(len(dtm)):
			'''
			if i > 2:
				dtm[j] = dtm[j]*int(avg_point_len)

			print(len(dtm[j]))
			'''
			song_points[j] += dtm[j]
			
	# return the list of data points representative of each point
	return song_points

# call the function to create data points for each song
all_songs = class1_songs + class2_songs
all_song_names = class1_song_names + class2_song_names
song_points = create_classification_points(class1_songs+class2_songs, class1_song_names+class2_song_names)

# seperate lists of data points for each class
class1_song_points = song_points[:len(class1_songs)]
class2_song_points = song_points[len(class1_songs):]

# list for predicted labels to be compared to truth labels
predictions = []
true_labels = [0]*len(class1_songs)+[1]*len(class2_songs)

# for each song in class 2, use SVM to determine if it best fits in class 1 or class 2 (should be class 2)
for i in range(len(all_songs)):
	# target point information
	point = song_points[i]
	name = all_song_names[i]
	if i < len(class1_songs):
		# copy of class 1 points, with the target removed
		class1_copy = class1_song_points[:]
		class1_copy.remove(point)

		# randomly omit one song from the second class of songs
		rand_omission = randint(0, len(class2_songs)-1)
		class2_copy = class2_song_points[:]
		class2_copy.remove(class2_copy[rand_omission])
		
		# training points is the combination of all points minus the target
		training_points = class1_copy + class2_copy
		
		# label class 1 songs as 0 and class 2 songs as 1 for training
		labels = ([0]*len(class1_copy)) + ([1]*len(class2_copy))
	else:
		# copy of class 2 points, with the target removed
		class2_copy = class2_song_points[:]
		class2_copy.remove(point)
		
		# randomly omit one song from the second class of songs
		rand_omission = randint(0, len(class1_songs)-1)
		class1_copy = class1_song_points[:]
		class1_copy.remove(class1_copy[rand_omission])
		
		# training points is the combination of all points minus the target
		training_points = class1_copy + class2_copy
	
		# label class 1 songs as 0 and class 2 songs as 1 for training
		labels = ([0]*len(class1_copy)) + ([1]*len(class2_copy))

	# Create multiple SVM variations
	h = 0.02
	C = 1.0

	svc = svm.SVC(kernel='linear', C=C).fit(training_points, labels)
	rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(training_points, labels)
	poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(training_points, labels)
	lin_svc = svm.SVC(C=C).fit(training_points, labels)

	# Predict the label based off of the SVMs above
	svc_predicted_label = svc.predict([point])[0]
	rbf_predicted_label = rbf_svc.predict([point])[0]
	poly_predicted_label = poly_svc.predict([point])[0]
	lin_predicted_label = lin_svc.predict([point])[0]

	predictions.append(svc_predicted_label)
	'''
	print('***'*10)
	print(svc_predicted_label)
	print(rbf_predicted_label)
	print(poly_predicted_label)
	print(lin_predicted_label)
	'''
	predicted_labels = [svc_predicted_label, rbf_predicted_label, poly_predicted_label, lin_predicted_label]

# Final output/results
print('Predicted Labels: ', predictions)
print('True Labels: ', true_labels)
print('Confusion Matrix: ', confusion_matrix(true_labels, predictions))
print('Classification Matrix: ', classification_report(true_labels, predictions))