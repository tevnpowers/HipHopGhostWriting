# import required modules
import os
import sys
import re
import operator
import numpy as np
from knn import find_knn

# loop through the given directory and get all song lyrics and names
def get_all_songs(directory):
	songs = []
	documents = []
	for filename in os.listdir("./"+directory):
		lyrics = parse_lyrics(filename, "./"+directory)
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
def get_ngrams(song, grams, n):
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
def get_words(song, dict, n):
	words = re.split('\\s+', song)
	for word in words:
		if word != '':
			if word not in dict:
				dict[word] = 1.0
			else:
				dict[word] += 1.0

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

# sort dictionary based on value
def sort_ngrams(grams):
	grams = sorted(grams.items(), key=operator.itemgetter(1), reverse=True)
	return grams

# normalize n grams to rates per 1000
def normalize_ngrams(grams):
	total = sum(grams.values())/1000.0
	for key in grams:
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