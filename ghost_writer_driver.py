from file_parser import *
from sklearn import svm

# check for correct parameters
if len(sys.argv) < 3:
	print('You must provide the following parameters: 1) Name of directory containing texts written by author one 2) Name of directory containing ghost written texts attributed to author one.')
	sys.exit()

# determine if the last parameter is a song or directory
if os.path.isdir(sys.argv[1]):
	class1_song_names, class1_songs = get_all_songs(sys.argv[1])
else:
	print('Unknown parameter ' + sys.argv[1] + '. Pleaes provide a file name or directory.')
	sys.exit()

# determine if the last parameter is a song or directory
if os.path.isdir(sys.argv[2]):
	class2_song_names, class2_songs = get_all_songs(sys.argv[2])
else:
	print('Unknown parameter ' + sys.argv[2] + '. Pleaes provide a file name or directory.')
	sys.exit()

# which features to extract
feature_list = [get_words, get_ngrams, get_word_grams, get_avg_word_length, get_avg_line_length]

# what n should be (only necessary for character gram and word gram extraction
n_list = [None, 4, 3, None, None,]

def create_classification_points(songs, song_names):
	song_points = []
	for song in songs:
		song_points.append([])

	for i in range(len(feature_list)):
		feature_func = feature_list[i]

		# build the representative points for each feature
		if feature_func in [get_song_length, get_avg_word_length, get_line_count, get_avg_line_length]:
			dtm, documents = build_single_feature(song_names, songs, feature_func)
		else:
			dtm, vocab, documents = build_feature_vocab(n_list[i], song_names, songs, feature_func)

		for j in range(len(dtm)):
			song_points[j] += dtm[j]

	return song_points

song_points = create_classification_points(class1_songs+class2_songs, class1_song_names+class2_song_names)
class1_song_points = song_points[:len(class1_songs)]
class2_song_points = song_points[len(class1_songs):]

for i in range(len(class2_songs)):
	point = class2_song_points[i]
	name = class2_song_names[i]
	class2_copy = class2_song_points[:]
	class2_copy.remove(point)

	training_points = class1_song_points + class2_copy
	labels = ([0]*len(class1_song_points)) + ([1]*len(class2_copy))

	h = 0.02
	C = 1.0
	
	svc = svm.SVC(kernel='linear', C=C).fit(training_points, labels)
	rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(training_points, labels)
	poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(training_points, labels)
	lin_svc = svm.SVC(C=C).fit(training_points, labels)
	
	svc_predicted_label = svc.predict([point])
	rbf_predicted_label = rbf_svc.predict([point])
	poly_predicted_label = poly_svc.predict([point])
	lin_predicted_label = lin_svc.predict([point])
	predicted_labels = [svc_predicted_label, rbf_predicted_label, poly_predicted_label, lin_predicted_label]
	print(predicted_labels)