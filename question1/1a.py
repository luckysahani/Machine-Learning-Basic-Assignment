import nltk
nltk.download('punkt') # for tokens
nltk.download("stopwords") # for stopwords
import random
import re
import time
from textblob import TextBlob
from nltk.corpus import stopwords
import os
from sklearn import cross_validation
from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
from sklearn.svm import SVC

output = ()
final_training_dataset = []
final_testing_dataset = []

def read_from_file(file_name_with_path,file_name):
	temp_dictionary = {}
	with open(file_name_with_path,"r") as lines:
		if file_name.startswith("spm"):
			for line in lines:
				if ( line == ""):
					continue
				tokens = nltk.word_tokenize(line)
				for token in tokens:
					if token.isalpha(): #and token not in stopwords.words():
						temp_dictionary.update({token: 'true'})
			return (temp_dictionary,"spam")					
		else:
			for line in lines:
				if ( line == ""):
					continue
				tokens = nltk.word_tokenize(line)
				for token in tokens:
					if token.isalpha(): #and token not in stopwords.words():
						temp_dictionary.update({token: 'true'})	
			return (temp_dictionary,"non_spam")


def traverse_over_files():
	global final_training_dataset
	number_of_dir_read = 0
	print "Your current working directory is :"+os.getcwd()
	dataset_directory = os.getcwd()+"/bare"
	print "Your dataset directory is :"+ dataset_directory
	get_list_of_all_subdirectories =  [dirs for root, dirs, files in os.walk(dataset_directory)]
	for dir_cur in get_list_of_all_subdirectories[0]:
		if(number_of_dir_read ==  9):
			test_directory = dataset_directory+"/"+dir_cur
			print "Current working for testing dataset on :"+test_directory
			number_of_dir_read = number_of_dir_read+1
			for root, dirs, files in os.walk(current_directory):
				for name in files:
					currentFile=os.path.join(root, name)
					# print "Reading from file "+str(currentFile)
					output = read_from_file(currentFile,name)
					final_testing_dataset.append(output)
		else:
			current_directory = dataset_directory+"/"+dir_cur
			print "Current working for training dataset on :"+current_directory
			number_of_dir_read = number_of_dir_read+1
			for root, dirs, files in os.walk(current_directory):
				for name in files:
					currentFile=os.path.join(root, name)
					# print "Reading from file "+str(currentFile)
					output = read_from_file(currentFile,name)
					final_training_dataset.append(output)

print "Reading files to make Dictionary"
start_time = time.time()
traverse_over_files()
end_time = time.time() - start_time
print "It took "+ str(end_time) + " to make the dicitonary"
print "Dictionary completed"
print '\nTraining data'
start_time = time.time()
# Training NLTK Naive Bayes Classifier
nltk_naivebayes_classifier = nltk.NaiveBayesClassifier.train(final_training_dataset)
# Training MultinomialNB Naive Bayes Classifier
MultinomialNB_classifier = SklearnClassifier(MultinomialNB()).train(final_training_dataset)
# Training BernoulliNB Naive Bayes Classifier
BernoulliNB_classifier = SklearnClassifier(BernoulliNB()).train(final_training_dataset)
end_time = time.time() - start_time
print "It took "+ str(end_time) + " to train the classifiers"
print 'Training Completed'


print '\nTesting data '
start_time = time.time()
match_nltk_naivebayes = 0
match_MultinomialNB = 0
match_BernoulliNB = 0
unmatch_nltk_naivebayes = 0
unmatch_MultinomialNB = 0
unmatch_BernoulliNB = 0
for data in final_testing_dataset:
	# NLTK Naive Bayes Classifier
	if( data[1] == nltk_naivebayes_classifier.classify(data[0])):
		match_nltk_naivebayes = match_nltk_naivebayes + 1
	else:
		unmatch_nltk_naivebayes = unmatch_nltk_naivebayes + 1
	# MultinomialNB
	if( data[1] == MultinomialNB_classifier.classify(data[0])):
		match_MultinomialNB = match_MultinomialNB + 1
	else:
		unmatch_MultinomialNB = unmatch_MultinomialNB + 1
	# BernoulliNB
	if( data[1] == BernoulliNB_classifier.classify(data[0])):
		match_BernoulliNB = match_BernoulliNB + 1
	else:
		unmatch_BernoulliNB = unmatch_BernoulliNB + 1
# Calculating Accuracy
nltk_naivebayes_accuracy = (float) (match_nltk_naivebayes )/ (match_nltk_naivebayes + unmatch_nltk_naivebayes)
MultinomialNB_accuracy = (float) (match_MultinomialNB ) / (match_MultinomialNB + unmatch_MultinomialNB)
BernoulliNB_accuracy = (float) (match_BernoulliNB)/ (match_BernoulliNB + unmatch_BernoulliNB)
BernoulliNB_accuracy_nltk = nltk.classify.accuracy(BernoulliNB_classifier, final_testing_dataset)
MultinomialNB_accuracy_nltk = nltk.classify.accuracy(MultinomialNB_classifier, final_testing_dataset)

end_time = time.time() - start_time
print "It took "+ str(end_time) + " to test the data "
print 'Testing Completed'

print '\nPrinting Accuracy\n'
print "BernoulliNB accuracy (using nltk accuracy funciton):"+ str(BernoulliNB_accuracy_nltk)
print "BernoulliNB accuracy (by calculation): "+ str(BernoulliNB_accuracy)
print "MultinomialNB accuracy (using nltk accuracy funciton):"+ str(MultinomialNB_accuracy_nltk)
print "MultinomialNB accuracy (by calculation): "+ str(MultinomialNB_accuracy)
print "NLTK Naive Bayes accuracy (by calculation): "+ str(nltk_naivebayes_accuracy)

print '\nPrinting Most informative features\n'
nltk_naivebayes_classifier.show_most_informative_features(5)
# See what to print and how
# see new nltk functions
# remove subject word
