import nltk
nltk.download('punkt') # for tokens
nltk.download("stopwords") # for stopwords
import re
from math import log
import time
from nltk.corpus import stopwords
import os
from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from nltk.stem import WordNetLemmatizer
word_lemmatizer = WordNetLemmatizer()
import sys
from sklearn.linear_model import Perceptron
# from sklearn.feature_extraction.text import CountVectorizer
# vectorizer = CountVectorizer(tokenizer=tokenize)

output = ()
main_dictionary = {}
final_training_dataset_values = []
final_training_dataset_keys = []
final_testing_dataset_values = []
final_testing_dataset_keys = []
testing_directory = sys.argv[1]
total_number_of_documents = 0
files_occured_dictionary = {}

def read_from_file_to_make_dictionary(file_name_with_path,file_name):
	to_check = {}
	with open(file_name_with_path,"r") as lines:
		for line in lines:
			if ( line == ""):
				continue
			tokens = nltk.word_tokenize(line)
			for token in tokens:
				token = token.lower() # making everything in lower case to avoid conflicts due to word case
				# if token.isalpha(): #and token not in stopwords.words():
				if token.isalpha() and token not in stopwords.words('english'):
					# main_dictionary.update({token: 0})

					lemmatized_word = word_lemmatizer.lemmatize(token)
					main_dictionary.update({lemmatized_word: 0})
					if( lemmatized_word not in to_check):
						if lemmatized_word in files_occured_dictionary:
							files_occured_dictionary[lemmatized_word] += 1
						else:
							files_occured_dictionary[lemmatized_word] = 1
						to_check[lemmatized_word] = False




def make_dictionary():
	global total_number_of_documents
	dataset_directory = os.getcwd()+"/bare"
	print "Your dataset directory is :"+ dataset_directory
	for root, dirs, files in os.walk(dataset_directory):
		for name in files:
			total_number_of_documents +=1
			read_from_file_to_make_dictionary(os.path.join(root, name),name)

def read_from_file(file_name_with_path,file_name):
	global total_number_of_documents
	temp_dictionary = main_dictionary.copy()
	token_to_be_updated = []
	with open(file_name_with_path,"r") as lines:
		for line in lines:
			if ( line == ""):
				continue
			tokens = nltk.word_tokenize(line)
			for token in tokens:
				token = token.lower() # making everything in lower case to avoid conflicts due to word case
				# if token.isalpha(): #and token not in stopwords.words():
				if token.isalpha() and token not in stopwords.words('english'):
					# temp_dictionary.update({token: 1})
					lemmatized_word = word_lemmatizer.lemmatize(token)
					temp_dictionary[lemmatized_word] += 1
					token_to_be_updated.append(lemmatized_word)
					# temp_dictionary.update({word_lemmatizer.lemmatize(token): 1})
		for token in list(set(token_to_be_updated)):
			# print temp_dictionary[token],token
			tf = 1 + log(temp_dictionary[token])
			idf = log( 1 + (total_number_of_documents/(1 + files_occured_dictionary[token])))
			# print "The number of times "+ str(token) +" has occured ="+str(files_occured_dictionary[token])+" and idf = "+str(idf)
			temp_dictionary[token]= idf*tf
		if file_name.startswith("spm"):
			return (list(temp_dictionary.values()),"spam")
		else:
			return (list(temp_dictionary.values()),"non_spam")


def traverse_over_files(testing_directory):
	# global final_training_dataset_values,final_training_dataset_keys
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
					final_testing_dataset_keys.append(output[0])
					final_testing_dataset_values.append(output[1])
		else:
			current_directory = dataset_directory+"/"+dir_cur
			print "Current working for training dataset on :"+current_directory
			for root, dirs, files in os.walk(current_directory):
				for name in files:
					currentFile=os.path.join(root, name)
					output = read_from_file(currentFile,name)
					# print output
					# raw_input()
					final_training_dataset_keys.append(output[0])
					final_training_dataset_values.append(output[1])

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

print '\nTraining data'
start_time = time.time()
perceptron_classifier = Perceptron()
perceptron_classifier.fit(final_training_dataset_keys, final_training_dataset_values)
end_time = time.time() - start_time
print "It took "+ str(end_time) + " to train the classifiers"
print 'Training Completed'

print '\nTesting data '
start_time = time.time()
# Calculating Accuracy
perceptron_classifier_accuracy = perceptron_classifier.score(final_testing_dataset_keys, final_testing_dataset_values)

end_time = time.time() - start_time
print "It took "+ str(end_time) + " to test the data "
print 'Testing Completed'

# print '\nprinting Accuracy'
print "\nCase "+str(testing_directory)+": Testing folder is part"+str(testing_directory)
print "-------------------------------------------------"
print "Perceptron accuracy : "+ str(perceptron_classifier_accuracy)


# print 'Training Size:'+str(len(final_training_dataset_keys))+' and Testing size = '+str(len(final_testing_dataset_keys))