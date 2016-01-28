import nltk
import random
import time
from textblob import TextBlob
from nltk.corpus import stopwords
import os
from sklearn import cross_validation

word_dicitonary = {}
test_directory =""

def read_from_file(file_name):
	with open(file_name,"r") as lines:
		for line in lines:
			if file_name.startswith("spm")

def traverse_over_files():
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
				# currentPath=os.path.abspath(name)
				# file_path = os.path.relpath(name)		
				print "Reading from file "+str(currentFile)
				read_from_file(currentFile)

print "Starting .."
start_time = time.time()
traverse_over_files()
end_time = time.time() - start_time
print "It took "+ str(end_time) + " to make the dicitonary"
print "Dictionary completed"
print 'Training data'
start_time = time.time()
# classifier = nltk.NaiveBayesClassifier.train(word_dicitonary)
end_time = time.time() - start_time
print "It took "+ str(end_time) + " to train the data for classifier"
print 'Training Completed'
print 'Testing data on :'+test_directory
start_time = time.time()
# See how
end_time = time.time() - start_time
print "It took "+ str(end_time) + " to test the data "
print 'Testing Completed'
print 'Printing Result'
# See what to print and how
# classifier.show_most_informative_features(5)
