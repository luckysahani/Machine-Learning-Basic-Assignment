import os
import struct
import time
from array import array as pyarray
from pylab import *
from numpy import *
from sklearn.neighbors import KNeighborsClassifier

nearest_neighours = sys.argv[1]

def process_labels_and_images(image_data,label_data,size,rows, cols):
    # digits=np.arange(10)
    indices = [ k for k in range(size) if label_data[k] in np.arange(10) ]
    N = len(indices)    
    images = zeros((N, rows*cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(indices)):
        images[i] = array(image_data[ indices[i]*rows*cols : (indices[i]+1)*rows*cols ])
        labels[i] = label_data[indices[i]]
    return images,labels

def load_data_from_mnist():
    global images_train,labels_train,images_test,labels_test,size,rows,cols
    path = os.getcwd()
    training_images_location = os.path.join(path, 'train-images.idx3-ubyte')
    training_labels_location = os.path.join(path, 'train-labels.idx1-ubyte')
    testing_images_location = os.path.join(path, 't10k-images.idx3-ubyte')
    testing_labels_location = os.path.join(path, 't10k-labels.idx1-ubyte')
    

    with open(training_labels_location,"r") as training_labels_file:
        magic_nr, size = struct.unpack(">II", training_labels_file.read(8))
        labels_train_data = pyarray("b", training_labels_file.read())

    print "Creating training images "
    start_time = time.time()
    with open(training_images_location,"r") as training_images_file:
        magic_nr, size, rows, cols = struct.unpack(">IIII", training_images_file.read(16))
        images_train_data = pyarray("b", training_images_file.read())
        # images_train = np.fromfile(training_images_file,dtype=np.uint8).reshape(len(labels_train),rows*cols)
    end_time = time.time() - start_time
    print "It took "+ str(end_time) + " to create the training images"

    labels_train,images_train = process_labels_and_images(images_train_data,labels_train_data,size,rows, cols)

    with open(testing_labels_location,"r") as testing_labels_file:
        magic_nr, size = struct.unpack(">II", testing_labels_file.read(8))
        labels_test_data = pyarray("b", testing_labels_file.read())
    

    print "Creating testing images "
    start_time = time.time()
    with open(testing_images_location,"r") as testing_images_file:
        magic_nr, size, rows, cols = struct.unpack(">IIII", testing_images_file.read(16))
        images_test_data = pyarray("b", testing_images_file.read())
        # images_test = np.fromfile(testing_images_file,dtype=np.uint8).reshape(len(labels_test),rows*cols)
    end_time = time.time() - start_time
    print "It took "+ str(end_time) + " to create the testing images"

    labels_test,images_test =process_labels_and_images(images_test_data,labels_test_data,size,rows, cols)
  


print "Creating Dataset from MNIST Data"
start_time = time.time()
load_data_from_mnist()
end_time = time.time() - start_time
print "It took "+ str(end_time) + " to make the dataset"

print '\nTraining data'
start_time = time.time()
knn_classifier = KNeighborsClassifier(n_neighbors=nearest_neighours)
knn_classifier.fit(images_train, labels_train) 
end_time = time.time() - start_time
print "It took "+ str(end_time) + " to train the classifier"
print 'Training Completed'

print '\nTesting data '
start_time = time.time()
match_knn_classifier = 0
unmatch_knn_classifier = 0
# print "Length of testing dataset : "+str(len(final_testing_dataset))
predicted_labels = knn_classifier.predict(images_test)
for i in range(0,len(images_test)):
    if( labels_test[i] == predicted_labels[i]):
        match_knn_classifier = match_knn_classifier + 1
    else:
        unmatch_knn_classifier = unmatch_knn_classifier + 1
# Calculating Accuracy
knn_classifier_accuracy = (float) (match_knn_classifier )/ (match_knn_classifier + unmatch_knn_classifier)
# knn_classifier_accuracy = knn_classifier.score(images_test, labels_test)

end_time = time.time() - start_time
print "It took "+ str(end_time) + " to test the data "
print 'Testing Completed'

print '\nprinting Accuracy'
print "\nCase "+str(nearest_neighours)+": Testing for n_neighbors = "+str(nearest_neighours)
print "-------------------------------------------------"
print "KNeighborsClassifier accuracy : "+ str(knn_classifier_accuracy)
