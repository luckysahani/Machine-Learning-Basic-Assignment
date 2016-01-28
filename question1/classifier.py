import nltk #basic nltk library
import re # To matchthe regular expressions
from math import sqrt,log #Basic math functions required (if)
import random #To randomize the data
import time #For time access
from textblob import TextBlob #For tokens
from nltk.corpus import stopwords #For stop words

stopwords   = nltk.corpus.stopwords.words('english')    #Words I should avoid since they have weak value for classification
my_file     = open("spam_files.txt", "r")   #my_file now has the name of each file that contains a spam email example
word        = {}    #a dictionary where I will storage all the words and which value they have (spam or ham)

for lines in my_file:   #for each name of file (which will be represenetd by LINES) of my_file
with open(lines.rsplit('\n')[0]) as email: #I will open the file pointed by LINES, and then, read the email example that is inside this file
    for phrase in email:    #After that, I will take every phrase of this email example I just opened
        try:    #and I'll try to tokenize it
            tokens = nltk.word_tokenize(phrase)
        except: 
            continue    #I will ignore non-ascii elements
        for c in tokens:    #for each token
            regex = re.compile('[^a-zA-Z]') #I will also exclude numbers
            c = regex.sub('', c)
            if (c): #If there is any element left
                if (c not in stopwords): #And if this element is a not a stopword
                    c.lower()
                    word.update({c: 'spam'})#I put this element in my dictionary. Since I'm analysing spam examples, variable C is labeled "spam".

my_file.close() 
email.close()

#The same logic is used for the Ham emails. Since my ham emails contain only ascii elements, I dont test it with TRY
my_file = open("ham_files.txt", "r")
for lines in my_file:
with open(lines.rsplit('\n')[0]) as email:
    for phrase in email:
        tokens = nltk.word_tokenize(phrase)
        for c in tokens:
            regex = re.compile('[^a-zA-Z]')
            c = regex.sub('', c)
            if (c):
                if (c not in stopwords):
                    c.lower()
                    word.update({c: 'ham'})

my_file.close() 
email.close()

#And here I train my classifier
classifier = nltk.NaiveBayesClassifier.train(word)
classifier.show_most_informative_features(5)