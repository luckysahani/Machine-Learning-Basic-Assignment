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

word_dictionary = {}
test_directory =""
output = ()
final_data = []

def read_from_file(file_name_with_path,file_name):
	temp_dictionary = {}
	with open(file_name_with_path,"r") as lines:
		if file_name.startswith("spm"):
			for line in lines:
				if ( line == ""):
					continue
				tokens = nltk.word_tokenize(line)
				for token in tokens:
					# print token.isalpha(),token
					if token.isalpha(): #and token not in stopwords.words():
						word_dictionary.update({token: 'spam'})
						temp_dictionary.update({token: 'true'})
			return (temp_dictionary,"spam")					
		else:
			for line in lines:
				if ( line == ""):
					continue
				tokens = nltk.word_tokenize(line)
				for token in tokens:
					# print token.isalpha(),token
					if token.isalpha(): #and token not in stopwords.words():
						word_dictionary.update({token: 'non_spam'})	
						temp_dictionary.update({token: 'true'})	
			return (temp_dictionary,"non_spam")


def traverse_over_files():
	global final_data
	number_of_dir_read = 0
	print "Your current working directory is :"+os.getcwd()
	dataset_directory = os.getcwd()+"/bare"
	print "Your dataset directory is :"+ dataset_directory
	get_list_of_all_subdirectories =  [dirs for root, dirs, files in os.walk(dataset_directory)]
	for dir_cur in get_list_of_all_subdirectories[0]:
		if(number_of_dir_read == 9):
			global test_directory
			test_directory = dataset_directory+"/"+dir_cur
			print "Testing directory is set to :"+test_directory
			break
		current_directory = dataset_directory+"/"+dir_cur
		print "Current directory is "+current_directory
		number_of_dir_read = number_of_dir_read+1
		for root, dirs, files in os.walk(current_directory):
			for name in files:
				currentFile=os.path.join(root, name)
				# print "Reading from file "+str(currentFile)
				output = read_from_file(currentFile,name)
				final_data.append(output)

print "Reading files to make Dictionary"
start_time = time.time()
traverse_over_files()
end_time = time.time() - start_time
print "It took "+ str(end_time) + " to make the dicitonary"
print "Dictionary completed"
print '\nTraining data'
start_time = time.time()
classifier = nltk.NaiveBayesClassifier.train(final_data)
end_time = time.time() - start_time
print "It took "+ str(end_time) + " to train the data for classifier"
print 'Training Completed'
print '\nTesting data on :'+test_directory
start_time = time.time()
# See how
end_time = time.time() - start_time
print "It took "+ str(end_time) + " to test the data "
print 'Testing Completed'
print '\nPrinting Result'
# See what to print and how
# classifier.show_most_informative_features(5)