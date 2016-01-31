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
from nltk.stem import WordNetLemmatizer
word_lemmatizer = WordNetLemmatizer()
from sklearn.svm import SVC
import sys

output = ()
main_dictionary = {}
final_training_dataset = []
final_testing_dataset = []
testing_directory = sys.argv[1]

def read_from_file_to_make_dictionary(file_name_with_path,file_name):
	with open(file_name_with_path,"r") as lines:
		for line in lines:
			if ( line == ""):
				continue
			tokens = nltk.word_tokenize(line)
			for token in tokens:
				token = token.lower() # making everything in lower case to avoid conflicts due to word case
				if token.isalpha() and token not in stopwords.words('english'):
					if(token != "Subject"):
						main_dictionary.update({word_lemmatizer.lemmatize(token): 0})



def make_dictionary():
	dataset_directory = os.getcwd()+"/bare"
	print "Your dataset directory is :"+ dataset_directory
	for root, dirs, files in os.walk(dataset_directory):
		for name in files:
			read_from_file_to_make_dictionary(os.path.join(root, name),name)

def read_from_file(file_name_with_path,file_name):
	temp_dictionary = {}
	with open(file_name_with_path,"r") as lines:
		for line in lines:
			if ( line == ""):
				continue
			tokens = nltk.word_tokenize(line)
			for token in tokens:
				token = token.lower() # making everything in lower case to avoid conflicts due to word case
				if token.isalpha() and token not in stopwords.words('english'):
					if(token != "Subject"):
						temp_dictionary.update({word_lemmatizer.lemmatize(token): 1})
		if file_name.startswith("spm"):
			return (temp_dictionary,"spam")
		else:
			return (temp_dictionary,"non_spam")


def traverse_over_files(testing_directory):
	global final_training_dataset
	print "Your current working directory is :"+os.getcwd()
	dataset_directory = os.getcwd()+"/bare"
	print "Your dataset directory is :"+ dataset_directory
	get_list_of_all_subdirectories =  [dirs for root, dirs, files in os.walk(dataset_directory)]
	for dir_cur in get_list_of_all_subdirectories[0]:
		if(dir_cur == "part"+ testing_directory):
			test_directory = dataset_directory+"/"+dir_cur
			print "Creating testing dataset for :"+test_directory
			for root, dirs, files in os.walk(test_directory):
				for name in files:
					currentFile=os.path.join(root, name)
					output = read_from_file(currentFile,name)
					final_testing_dataset.append(output)
		else:
			current_directory = dataset_directory+"/"+dir_cur
			print "Current working for training dataset on :"+current_directory
			for root, dirs, files in os.walk(current_directory):
				for name in files:
					currentFile=os.path.join(root, name)
					output = read_from_file(currentFile,name)
					final_training_dataset.append(output)

print "Reading files to make Dictionary "
start_time = time.time()
make_dictionary()
end_time = time.time() - start_time
print "It took "+ str(end_time) + " to make the dicitonary"
print "Dictionary completed"

print "Reading files to make Training Dataset "
start_time = time.time()
traverse_over_files(str(testing_directory))
end_time = time.time() - start_time
print "It took "+ str(end_time) + " to make the Training Dataset"
print "Training Dataset completed"

# print '\nTraining data'
# start_time = time.time()
# # Training NLTK Naive Bayes Classifier
# nltk_naivebayes_classifier = nltk.NaiveBayesClassifier.train(final_training_dataset)
# # Training MultinomialNB Naive Bayes Classifier
# MultinomialNB_classifier = SklearnClassifier(MultinomialNB()).train(final_training_dataset)
# # Training BernoulliNB Naive Bayes Classifier
# BernoulliNB_classifier = SklearnClassifier(BernoulliNB()).train(final_training_dataset)
# end_time = time.time() - start_time
# print "It took "+ str(end_time) + " to train the classifiers"
# print 'Training Completed'

# print '\nTesting data '
# start_time = time.time()
# match_nltk_naivebayes = 0
# unmatch_nltk_naivebayes = 0
# for data in final_testing_dataset:
# 	if( data[1] == nltk_naivebayes_classifier.classify(data[0])):
# 		match_nltk_naivebayes = match_nltk_naivebayes + 1
# 	else:
# 		unmatch_nltk_naivebayes = unmatch_nltk_naivebayes + 1
# # Calculating Accuracy
# nltk_naivebayes_accuracy = (float) (match_nltk_naivebayes )/ (match_nltk_naivebayes + unmatch_nltk_naivebayes)
# BernoulliNB_accuracy = nltk.classify.accuracy(BernoulliNB_classifier, final_testing_dataset)
# MultinomialNB_accuracy = nltk.classify.accuracy(MultinomialNB_classifier, final_testing_dataset)

# end_time = time.time() - start_time
# print "It took "+ str(end_time) + " to test the data "
# print 'Testing Completed'

# print '\nprinting Accuracy'
# print "\nCase "+str(testing_directory)+": Testing folder is part"+str(testing_directory)
# print "-------------------------------------------------"
# print "BernoulliNB accuracy : "+ str(BernoulliNB_accuracy)
# print "MultinomialNB accuracy: "+ str(MultinomialNB_accuracy)
# print "NLTK Naive Bayes accuracy (by calculation): "+ str(nltk_naivebayes_accuracy)
# nltk_naivebayes_classifier.show_most_informative_features(5)
