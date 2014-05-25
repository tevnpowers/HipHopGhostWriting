# import required modules
import os
import sys
import re
import operator
import nltk
import numpy as np
from knn import find_knn
from collections import Counter

CHARACTER_N_GRAMS = 4

# simplified tagset mapping (from nltk.tag.simplify source)
wsj_mapping = {'-lrb-': '(',   '-rrb-': ')',    '-lsb-': '(', '-rsb-': ')',   '-lcb-': '(',    '-rcb-': ')',
				'-none-': '',   'cc': 'CNJ',     'cd': 'NUM',
				'dt': 'DET',    'ex': 'EX',      'fw': 'FW', # existential "there", foreign word 
				'in': 'P',      'jj': 'ADJ',     'jjr': 'ADJ',
				'jjs': 'ADJ',   'ls': 'L',       'md': 'MOD',  # list item marker 
				'nn': 'N',      'nnp': 'NP',     'nnps': 'NP',
				'nns': 'N',     'pdt': 'DET',    'pos': '',
				'prp': 'PRO',   'prp$': 'PRO',   'rb': 'ADV',
				'rbr': 'ADV',   'rbs': 'ADV',    'rp': 'PRO',
				'sym': 'S',     'to': 'TO',      'uh': 'UH',
				'vb': 'V',      'vbd': 'VD',     'vbg': 'VG',
				'vbn': 'VN',    'vbp': 'V',      'vbz': 'V',
				'wdt': 'WH',    'wp': 'WH',      'wp$': 'WH',
				'wrb': 'WH',
				'bes': 'V',     'hvs': 'V',     'prp^vbp': 'PRO'   # additions for NPS Chat corpus 
				}

wsj_to_description = {
				'ADJ': 'Adjective',
				'ADV': 'Adverb',
				'CNJ': 'Conjuction',
				'DET': 'Determiner',
				'EX': 'Existential',
				'FW': 'Foreign word',
				'MOD': 'Modal Verb',
				'N': 'Noun',
				'NP': 'Proper Noun',
				'NUM': 'Number',
				'PRO': 'Pronoun',
				'P': 'Preposition',
				'TO': 'The word \'to\'',
				'UH': 'Interjectoin',
				'V': 'Verb',
				'VD': 'Past Tense Verb',
				'VG': 'Present Participle Verb',
				'VN': 'Past Participle Verb',
				'WH': 'wh Determiner'
				}

# loop through the given directory and get all song lyrics and names
def get_all_songs(directory):
	songs = []
	documents = []
	for filename in os.listdir("./"+directory):
		lyrics = parse_lyrics(filename, "./" + directory)
		documents.append(filename)
		songs.append(lyrics)
	return (documents, songs)
	
# extract just the letters, numbers and whitespace from the song
def parse_lyrics(filename, directory):
	file = open(directory+"/"+filename)
	lyrics = file.read().lower()
	lyrics = "".join(re.findall("[a-zA-Z0-9 \n]+", lyrics))
	return lyrics

# create a dictionary of character n grams
def get_character_grams(song, grams, n):
	for i in range(len(song)):
		gram = song[i:i+n]
		if gram not in grams:
			grams[gram] = 1.0
		else:
			grams[gram] += 1.0

# create a dictionary of word n grams
def get_word_grams(song, grams, n):
	words = re.split('\\s+', song)
	for i in range(len(words)):
		gram = tuple(words[i:i+n])
		if gram not in grams:
			grams[gram] = 1.0
		else:
			grams[gram] += 1.0

# create a dictionary of words
def get_top_words(song, dict, n):
	words = re.split('\\s+', song)
	word_counts = Counter(words)
	top_words = word_counts.most_common(n)
	for key, value in top_words:
		dict[key] = float(value)

# create a dictionary of word n grams
def get_pos(song, dict, n):
	tokens = nltk.word_tokenize(song)
	pos = [tag for (word, tag) in nltk.pos_tag(tokens)]

	pos_counts = Counter(pos)
	top_pos = pos_counts.most_common(n)
	for key, value in top_pos:
		dict[key] = float(value)

# get the number of characters in the song
def get_song_length(song):
	return len(song)

# return the average word length in the song
def get_avg_word_length(song):
	words = re.split('\\s+', song)
	chars = 0.0
	for word in words:
		chars += len(word)
	return chars/len(words)

# get the number of lines in the song
def get_line_count(song):
	lines = re.split('\n', song)
	return len(lines)

# get the average length of lines in the song
def get_avg_line_length(song):
	lines = re.split('\n', song)
	total_length = 0.0
	for line in lines:
		total_length += len(line)
	return total_length/len(lines)

def get_word_density(song):
	words = re.split('\\s+', song)
	unique = set(words)
	return len(unique)/len(words)
	
def get_pos_density(song):
	tokens = nltk.word_tokenize(song)
	all_pos = [pos for (word, pos) in nltk.pos_tag(tokens)]
	unique_pos = set(all_pos)
	return len(unique_pos)/len(all_pos)
	
# create a dictionary of character n grams
def get_character_gram_density(song):
	n = CHARACTER_N_GRAMS
	grams = []
	for i in range(len(song)):
		gram = song[i:i+n]
		grams.append(gram)

	unique_grams = set(grams)
	return len(unique_grams)/len(grams)

# normalize n grams to rates per 1000
def normalize_ngrams(grams):
	total = sum(grams.values())/10000000.0
	for key in grams:
		#rint(grams[key]/total, grams[key]/(total/1000.0))
		grams[key] /= total

# create a vocab dictionary
def create_vocab(dictionaries):
	vocab = set()
	for dict in dictionaries:
		for key in dict.keys():
			vocab.add(key)
	return np.array(list(vocab))

# build a vocabulary across all documents
def build_feature_vocab(n, song_names, songs, feature_func):
	dtm = []
	dictionaries = []

	# create a vocab for each document
	for doc in songs:
		doc_grams = {} 
		doc_vocab = feature_func(doc, doc_grams,n)
		normalize_ngrams(doc_grams)
		dictionaries.append(doc_grams)

	# merge all documents into a single vocabulary
	vocab = create_vocab(dictionaries)

	# update each document to have a dictionary representing all documents
	dtm = []
	for dict in dictionaries:
		rates = []
		for gram in vocab:
			if gram in dict:
				rates.append(dict[gram])
			else:
				rates.append(0.0)
		dtm.append(rates)
	#dtm = np.array(dtm)
	
	return dtm, vocab, np.array(song_names)

# build an array of values for features that have a single value (not a high dimensional point)
def build_single_feature(song_names, songs, feature_func):
	values = []
	for doc in songs:
		value = feature_func(doc)
		values.append([value])

	#dtm = np.array(values)
	dtm = values
	return dtm, np.array(song_names)