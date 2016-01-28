#CS771: Machine learning: tools, techniques, applications

1. Consider the spam data set available on the course ftp site in the directory ’assgnData’, file
spam.tar.gz. It has 10 sub-directories containing spam and non-spam mails, along with the subject.
The spam mails have file names starting with ’spm’. The data set is not balanced. Spam mails
are much fewer than non-spam ones. Use the naive Bayes algorithm to build a classifier to classify
mail as spam or non-spam. Report 10-fold cross validated results using nine sub-directories for
training and the tenth one for testing in each instance.
(a) First do it on the mail subject+body as is.
(b) Then repeat after stop words (fluff or non-content words) have been removed.
(c) Repeat a third time after removing stop words and lemmatizing the email. This means
retaining only the lemma or root of each word. So, for example inflectional markers (e.g.
tense) for a verb will be lemmatized to just the root verb.
Stop word lists and lemmatizers are available on the internet.

2. Build a spam/non-spam classifier using a linear discriminant function method using a bag of words
(BoW) representation in three different ways:
(a) Use a binary BoW representation. Value is 1 if the word occurs in the subject/body of the
mail, 0 if it does not.
(b) Use a term frequency based BoW representation. Term frequency counts the number of times
a word occurs in the document (normalized by the length of the document).
(c) Use the tf-idf BoW representation.
Again you should report 10-fold cross validated results.
Note on text representations: Vector space representations for text define a vector space
with dimension |V|, where V is the vocabulary for the documents (that is the set of words in the
documents). There are different ways of encoding a document as a vector. The common ones are:
1. Binary frequency: The entry corresponding to the word in the vector is 1 if the word occurs
in the document and 0 otherwise.
2. Term frequency tf (t, d): Let the raw frequency of the term t in document d be f (t, d). There
are numerous definitions for tf (t, d) using f (t, d).
3. Inverse document frequency idf (t, D): If a word (or term) occurs in every document of a
collection of documents D then it does not contain much information about the document
(e.g. the word ’the’). To correct for this the inverse document frequency idf (t, D) is defined.

The entry for term t in the vector for document d is given by tf (t, d) × idf (t, D). This is high
whenever the term is frequent in d but occurs infrequently in D.
The vocabulary V for a collection of documents is usually constructed from the set of all words
present in D after the stop words have been removed and lemmatization has been done. So, a
vector contains essentially content words. Note that the BoW representation does not take order
of word occurrence into account - that is the reason it is called a bag. A query document may
contain words not in V. These words are just ignored since the query document must also be
represented in the same vector space.

3. Build a k-NN classifier for the MNIST data set available on the ftp site in directory ’assgnData’.
This data consists of images of handwritten numerals from 0 to 9. There are 4 files. Training and
test data are separate and the images and labels are in separate files giving a total of 4 files. You
should use the training data as the model and the test data for testing. Test with different values
of k (say 1 to 4) to see which one works best. Also, try with different metrics - at least 3 different
metrics.

