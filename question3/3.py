#!/usr/bin/python
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.data import loadlocal_mnist
import os
import time

def KNN_classifier(nearest_neighours = 3, metric = 'manhattan'):
    # print "Creating Dataset from MNIST Data"
    start_time = time.time()
    training_image_data, training_label_data = loadlocal_mnist(
        images_path=os.getcwd()+'/train-images.idx3-ubyte', 
        labels_path=os.getcwd()+'/train-labels.idx1-ubyte')
    testing_image_data, testing_label_data = loadlocal_mnist(
        images_path=os.getcwd()+'/t10k-images.idx3-ubyte', 
        labels_path=os.getcwd()+'/t10k-labels.idx1-ubyte')
    end_time = time.time() - start_time
    # print "It took "+ str(end_time) + " to make the dataset"

    # print '\nTraining data'
    start_time = time.time()
    knn_classifier = KNeighborsClassifier(n_neighbors=nearest_neighours, metric=metric)
    knn_classifier.fit(training_image_data, training_label_data) 
    end_time = time.time() - start_time
    # print "It took "+ str(end_time) + " to train the classifier"
    # print 'Training Completed'

    # print '\nTesting data '
    start_time = time.time()
    match_knn_classifier = 0
    unmatch_knn_classifier = 0
    predicted_labels = knn_classifier.predict(testing_image_data)
    for i in range(0,len(testing_image_data)):
        if( testing_label_data[i] == predicted_labels[i]):
            match_knn_classifier = match_knn_classifier + 1
        else:
            unmatch_knn_classifier = unmatch_knn_classifier + 1
    knn_classifier_accuracy = (float) (match_knn_classifier )/ (match_knn_classifier + unmatch_knn_classifier)
    # knn_classifier_accuracy = knn_classifier.score(images_test, labels_test)
    end_time = time.time() - start_time
    # print "It took "+ str(end_time) + " to test the data "

    # print '\nPrinting Accuracy'
    print "\nTesting for n_neighbors = "+str(nearest_neighours)+" and metric = "+str(metric)
    print "-------------------------------------------------"
    print "KNeighborsClassifier accuracy : "+ str(knn_classifier_accuracy)

    return knn_classifier_accuracy

if __name__ == '__main__':
    sum_of_accuracy_for_a_metric = 0.0
    metrics = ['euclidean', 'manhattan', 'minkowski']
    for metric in metrics:
        sum_of_accuracy_for_a_metric = 0.0
        for k in range(1,5):
            sum_of_accuracy_for_a_metric += KNN_classifier(k, metric)
        print "\nMean Accuracy for metric : "+str(metric)+" is "+str(sum_of_accuracy_for_a_metric/4)
