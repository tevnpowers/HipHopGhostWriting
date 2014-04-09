from file_parser import *

# COMMENTS FOR THIS PROGRAM ARE THE SAME AS authorship.py EXCEPT THE PARAMETERS ARE DIFFERENT (explained in the paper)

if len(sys.argv) < 3:
	print('You must provide the following parameters: 1) Name of directory containing texts written by author one 2) Name of directory containing texs written by author two 3) Name of the file containing the disputed text.')
	sys.exit()

author1 = sys.argv[1]
author1_song_names, author1_songs = get_all_songs(author1)

author2 = sys.argv[2]
author2_song_names, author2_songs = get_all_songs(author2)

all_songs = author1_songs + author2_songs
all_song_names = author1_song_names + author2_song_names

feature_list = [get_words, get_ngrams, get_word_grams, get_song_length, get_avg_word_length, get_line_count, get_avg_line_length]
feature_accuracy = [0.0]*len(feature_list)
n_list = [None, 3, 2, None, None, None, None]
knn_list = [5, 5, 5, 5, 5, 5, 5, 5]

correct = 0.0
for j in range(len(author2_songs)):
	song = author2_songs[j]
	song_name = author2_song_names[j]
	song_index = all_songs.index(song)
	song_name_index = all_song_names.index(song_name)
	all_songs[0], all_songs[song_index] = all_songs[song_index], all_songs[0] 
	all_song_names[0], all_song_names[song_name_index] = all_song_names[song_name_index], all_song_names[0] 
	
	total_author1_votes = 0
	total_author2_votes = 0

	for i in range(len(feature_list)):
		feature_func = feature_list[i]

		if feature_func in [get_song_length, get_avg_word_length, get_line_count, get_avg_line_length]:
			dtm, documents = build_single_feature(all_song_names, all_songs, feature_func)
		else:
			dtm, vocab, documents = build_feature_vocab(n_list[i], all_song_names, all_songs, feature_func)
			#print('vocab:', vocab)
			
		#print('dtm:', dtm)
		#print('documents:', documents)
		#sys.exit()

		neighbors = documents[1:]
		k = knn_list[i]
		closest_indexes = find_knn(k, dtm[0], dtm[1:])
		author1_feature_votes = 0
		author2_feature_votes = 0
		for index in closest_indexes:
			if neighbors[index] in author1_song_names:
				#print('Author 1:', neighbors[index])
				author1_feature_votes += 1
			elif neighbors[index] in author2_song_names:
				#print('Author 2:', neighbors[index])
				author2_feature_votes += 1
			else:
				print('Something unexpected happened...')

		if author1_feature_votes > author2_feature_votes:
			print('Author of', song_name, 'is', author1, 'according to feature', feature_func.__name__)
			total_author1_votes += 1
		elif author2_feature_votes > author1_feature_votes:
			print('Author of', song_name, 'is', author2, 'according to feature', feature_func.__name__)
			total_author2_votes += 1
			feature_accuracy[i] += 1.0
		else:
			print('Looks like we have a tie for', feature_func.__name__)

	if total_author1_votes > total_author2_votes:
		print('Author of', song_name, 'is', author1, 'according to the majority vote system!\n')
	elif total_author2_votes > total_author1_votes:
		print('Author of', song_name, 'is', author2, 'according to the majority vote system!\n')
		correct += 1.0
	else:
		print('Looks like we have a tie in our system!\n')

print('Percentage correct: ', correct/len(author2_songs))
print('Feature Accuracy:', feature_accuracy)
feature_accuracy = [val/len(author2_songs) for val in feature_accuracy]
print('Feature Accuracy:', feature_accuracy)