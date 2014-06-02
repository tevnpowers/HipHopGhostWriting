# In this example we are going to create a simple HTML
# page with 2 input fields (numbers), and a link.
# Using jQuery we are going to send the content of both
# fields to a route on our application, which will
# sum up both numbers and return the result.
# Again using jQuery we'l show the result on the page


# We'll render HTML templates and access data sent by GET
# using the request object from flask. jsonigy is required
# to send JSON as a response of a request
from flask import Flask, render_template, request, jsonify, url_for

# Libraries for algorithm
from feature_extraction import *
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import RFE, SelectKBest, chi2
from random import randint

# Initialize the Flask application
app = Flask(__name__)


feature_list = [get_line_count, get_avg_word_length, get_avg_line_length, get_word_density, get_pos_density, get_character_gram_density, get_top_words, get_pos]

feature_func_names = [get_line_count.__name__, get_avg_word_length.__name__, get_avg_line_length.__name__, get_word_density.__name__, get_pos_density.__name__, get_character_gram_density.__name__, get_top_words.__name__, get_pos.__name__]

n_list = [None, None, None, None, None, None, 100, 10]

class1_song_points = []
class2_song_points = []
word_vocab = []
feature_names = []

@app.route('/extract_features')
def extract_features():
	global class1_song_points, class2_song_points, word_vocab, feature_names

	print('Extracting features from songs...')

	artist1 = request.args.get('artist1', 0, type=str)
	artist2 = request.args.get('artist2', 0, type=str)

	class1_song_names, class1_songs = get_all_songs(artist1)

	class2_song_names, class2_songs = get_all_songs(artist2)

	feature_names = ['Line Count', 'Average Word Length', 'Average Line Length', 'Word Density', 'POS Density', 'Character 4-Gram Density']
	word_vocab = []

	# call the function to create data points for each song
	all_songs = class1_songs + class2_songs
	all_song_names = class1_song_names + class2_song_names

	songs = class1_songs + class2_songs
	song_names = class1_song_names + class2_song_names

	# list to contain the data point of each song
	song_points = []

	# each data point is represented as a list (the list will dynamically grow with rates from each feature)
	for song in songs:
		song_points.append([])

	# loop through each feature function in the given list, adding the feature values to the data point for each song
	for i in range(len(feature_list)):
		feature_func = feature_list[i]

		# build the representative points for each feature (the dtm holds the feature data for each song)
		if feature_func in [get_song_length, get_avg_word_length, get_line_count, get_avg_line_length, get_word_density, get_pos_density, get_character_gram_density]:
			dtm, documents = build_single_feature(song_names, songs, feature_func)
		else:
			dtm, vocab, documents = build_feature_vocab(n_list[i], song_names, songs, feature_func)

		if feature_func == get_top_words or feature_func == get_pos:
			feature_names += vocab
		if feature_func == get_top_words:
			word_vocab += vocab

		# add the values for this feature set to the high dimensional point for each song
		for j in range(len(dtm)):
			song_points[j] += dtm[j]

	# seperate lists of data points for each class
	class1_song_points = song_points[:len(class1_songs)]
	class2_song_points = song_points[len(class1_songs):]
	return jsonify(value=True)

@app.route('/discriminatory_features')
def discriminatory_features():
	print('Finding most discriminatory features...')

	NUM_FEATURES = 10

	all_points = class1_song_points + class2_song_points
	true_labels = [0]*len(class1_song_points)+[1]*len(class2_song_points)

	feature_indices = []
	for i in range(NUM_FEATURES):
		selector = SelectKBest(chi2, i+1)
		selector.fit(all_points, true_labels)

		new_indices = selector.get_support(indices=True)
		for index in new_indices:
			if index not in feature_indices:
				feature_indices.append(index)

	feature_descriptions = []

	for index in feature_indices:
		feature = feature_names[index]
		if feature.lower() in wsj_mapping.keys():
			key = wsj_mapping[feature.lower()]
			description = key + ': ' + wsj_to_description[key]
		elif feature in word_vocab:
			description = 'The word: ' + feature
		else:
			description = feature
		feature_descriptions.append(description)
	return jsonify(features=feature_descriptions)

@app.route('/algorithm_confidence')
def algorithm_confidence():
	print('Computing algorithm confidence...')

	song_points = class1_song_points + class2_song_points
	true_labels = [0]*len(class1_song_points)+[1]*len(class2_song_points)
	
	svc_predictions = []
	# for each song in class 2, use SVM to determine if it best fits in class 1 or class 2 (should be class 2)
	for i in range(len(song_points)):
		# target point information
		point = song_points[i]
		if i < len(class1_song_points):
			# copy of class 1 points, with the target removed
			class1_copy = class1_song_points[:]
			class1_copy.remove(point)

			# training points is the combination of all points minus the target
			all_points = class1_copy + class2_song_points
			
			# label class 1 songs as 0 and class 2 songs as 1 for training
			labels = ([0]*len(class1_copy)) + ([1]*len(class2_song_points))
		else:
			# copy of class 2 points, with the target removed
			class2_copy = class2_song_points[:]
			class2_copy.remove(point)

			# training points is the combination of all points minus the target
			all_points = class1_song_points + class2_copy
		
			# label class 1 songs as 0 and class 2 songs as 1 for training
			labels = ([0]*len(class1_song_points)) + ([1]*len(class2_copy))

		# Create multiple SVM variations
		svc = svm.SVC(kernel='linear').fit(all_points, labels)

		# Predict the label based off of the SVMs above
		svc_predicted_label = svc.predict([point])[0]
		svc_predictions.append(svc_predicted_label)

	correct = 0.0
	for i in range(len(svc_predictions)):
		if svc_predictions[i] == true_labels[i]:
			correct += 1.0

	print('Confusion Matrix: ', confusion_matrix(true_labels, svc_predictions))
	print('Classification Matrix: ', classification_report(true_labels, svc_predictions))
	
	accuracy = float(correct*100/len(svc_predictions))
	print('Percent Accuracy: ' + str(accuracy) + '%')
	print('Program complete...')
	return jsonify(confidence=str(int(accuracy))+'%')

# Route that will process the AJAX request, sum up two
# integer numbers (defaulted to zero) and return the
# result as a proper JSON response (Content-Type, etc.)
@app.route('/_add_numbers')
def add_numbers():
    a = request.args.get('a', 0, type=int)
    b = request.args.get('b', 0, type=int)
    return jsonify(result=a + b)

# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation
@app.route('/')
def index():
	artists = ['Jay_Z', 'Will_Smith']
	return render_template('index.html', artists=artists)

if __name__ == '__main__':
    app.run(
        debug=True
    )