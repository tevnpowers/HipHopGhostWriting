from file_parser import *

# check for correct parameters
if len(sys.argv) < 4:
	print('You must provide the following parameters: 1) Name of directory containing texts written by author one 2) Name of directory containing texs written by author two 3) Name of the file containing the disputed text.')
	sys.exit()

# determine if the last parameter is a song or directory
if os.path.isfile(sys.argv[3]):
	songs = [parse_lyrics(sys.argv[3], '.')]
	song_names = [sys.argv[3]]
elif os.path.isdir(sys.argv[3]):
	song_names, songs = get_all_songs(sys.argv[3])
else:
	print('Unknown parameter ', sys.argv[3], '. Pleaes provide a file name or directory.')
	sys.exit()

# get all the songs for both writers
author1 = sys.argv[1]
author1_song_names, author1_songs = get_all_songs(author1)
author2 = sys.argv[2]
author2_song_names, author2_songs = get_all_songs(author2)

# which features to extract
feature_list = [get_words, get_ngrams, get_word_grams, get_song_length, get_avg_word_length, get_line_count, get_avg_line_length]

# what n should be (only necessary for character gram and word gram extraction
n_list = [None, 4, 3, 2, None, None, None, None]

# what k should be for the KNN algo
knn_list = [5, 5, 5, 5, 5, 5, 5, 5]

# loop through each document in the list of songs to be inspected
for j in range(len(songs)):
	song = songs[j]
	song_name = song_names[j]
	
	all_songs = [song] + author1_songs + author2_songs
	all_song_names = [song_name] + author1_song_names + author2_song_names

	total_author1_votes = 0
	total_author2_votes = 0

	# loop through each feature to be extracted
	for i in range(len(feature_list)):
		feature_func = feature_list[i]

		# build the representative points for each feature
		if feature_func in [get_song_length, get_avg_word_length, get_line_count, get_avg_line_length]:
			dtm, documents = build_single_feature(all_song_names, all_songs, feature_func)
		else:
			dtm, vocab, documents = build_feature_vocab(n_list[i], all_song_names, all_songs, feature_func)
			
		# run KNN
		neighbors = documents[1:]
		k = knn_list[i]
		closest_indexes = find_knn(k, dtm[0], dtm[1:])

		# tally up author 1 and author 2 "close" points
		author1_feature_votes = 0
		author2_feature_votes = 0
		for index in closest_indexes:
			if neighbors[index] in author1_song_names:
				author1_feature_votes += 1
			elif neighbors[index] in author2_song_names:
				author2_feature_votes += 1
			else:
				print('Something unexpected happened...')

		# figure out if author 1 or author 2 had more "close" points, tally up
		if author1_feature_votes > author2_feature_votes:
			print('Author of', song_name, 'is', author1, 'according to feature', feature_func.__name__)
			total_author1_votes += 1
		elif author2_feature_votes > author1_feature_votes:
			print('Author of', song_name, 'is', author2, 'according to feature', feature_func.__name__)
			total_author2_votes += 1
		else:
			print('Looks like we have a tie for', feature_func.__name__)

	# count final votes, to see who wins
	if total_author1_votes > total_author2_votes:
		print('Author of', song_name, 'is', author1, 'according to the majority vote system!\n')
	elif total_author2_votes > total_author1_votes:
		print('Author of', song_name, 'is', author2, 'according to the majority vote system!\n')
	else:
		print('Looks like we have a tie in our system!\n')