# CE4041 Project 3
# HOG SVM Classifier
# Reece Gavin ID: 17197589

# Import all necessary libraries
from skimage.feature import hog
from sklearn.svm import LinearSVC
from scipy import ndimage
import numpy as np
import pickle
import sys
import itertools
import threading
import time
import sys

# This section of code allows the hog-svm-train-dataset.64x64x3.npz to be read in from the terminal
# The code accepts an argument from the terminal and stores it in variable training_data
# A variable npz_file is used to load the input training_data
training_data = sys.argv[1]
npz_file = np.load(training_data)
list(npz_file.keys())

print("Please wait, this may take some time..")

# Array is created with strings 'posex' and 'negex'
['posex', 'negex']

# Variable posex is equal to the posex of variable npz_file and likewise the same with negex
posex = npz_file['posex']
negex = npz_file['negex']

# The positive labels are stored in an array called positive_labels and the same for the negative labels
positive_labels = np.full((95, 1), 1)
negative_labels = np.full((984, 1), -1)

# The positive_labels and negative_labels are concatenated into one array called all_labels
all_labels = np.concatenate((positive_labels, negative_labels))

# Numpy ravel is used to return a contiguous flattened array.
all_labels = np.ravel(all_labels)


# A function called HogSvm_Classifier is created which accepts inputs: scale, hog_length, npz_file, all_labels, posex, negex, pixels
def HogSvm_Classifier(scale, hog_length, npz_file, all_labels, posex, negex, pixels):

    # The training images are reshaped before using them to train the classifier
    posex = ndimage.zoom(posex, [1.0, scale, scale, 1.0])
    negex = ndimage.zoom(negex, [1.0, scale, scale, 1.0])

    # An array called positive_hogs is initialised with having a size 95 x hog_length where hog_length is 1764
    positive_hogs = np.zeros(shape=(95, hog_length))  # 1764

    # A for loop is used to create each positive hog and store it in the positive_hogs array
    # At the end of the for loop, there will be an array which contains 95 positive hogs, all with length 1764
    # he shape of the positive_hogs array is then printed
    for i in range(1, 94):
        positive_hogs[i] = hog(posex[i],
                               orientations=9,
                               pixels_per_cell=(pixels, pixels),
                               cells_per_block=(2, 2),
                               visualize=False,
                               multichannel=True)
    print("Positive_hogs shape", positive_hogs.shape)

    # An array called negative_hogs is initialised with having a size 984 x hog_length where hog_length is 1764
    negative_hogs = np.zeros(shape=(984, hog_length))

    # A for loop is used to create each negative hog and store it in the negative_hogs array
    # At the end of the for loop, there will be an array which contains 984 negative hogs, all with length 1764
    # The shape of the negative_hogs array is then printed
    for i in range(1, 983):
        negative_hogs[i] = hog(negex[i],
                               orientations=9,
                               pixels_per_cell=(pixels, pixels),
                               cells_per_block=(2, 2),
                               visualize=False,
                               multichannel=True)
    print("Negative_hogs shape", negative_hogs.shape)

    # The positive_hogs and negative_hogs are concatenated into a single array called all_hogs and the shape of
    # all_hogs is printed
    all_hogs = np.concatenate((positive_hogs, negative_hogs))
    print("all_hogs shape", all_hogs.shape)

    # A variable "classifier" is used to initiate LinearSVC()
    classifier = LinearSVC()

    # The classifier is trained using all_hogs ans all_labels
    classifier.fit(all_hogs, all_labels)

    return classifier



# 4 distinct classifiers are created

# Classifier_One is based for an image 64x64 pixels, has a hog length of 1764, and has 8 pixels per cell
classifier_One = HogSvm_Classifier(1, 1764, npz_file, all_labels, posex, negex, 8)

# Classifier_Two is based for an image 48x48 pixels (Scale= 0.75 of 64), has a hog length of 1764, and has 6 pixels per cell
classifier_Two = HogSvm_Classifier(0.75, 1764, npz_file, all_labels, posex, negex, 6)

# Classifier_Three is based for an image 40x40 pixels (Scale= 0.625 of 64), has a hog length of 2916, and has 4 pixels per cell
classifier_Three = HogSvm_Classifier(0.625, 2916, npz_file, all_labels, posex, negex, 4)

# Classifier_Four is based for an image 32x32 pixels (Scale= 0.5 of 64), has a hog length of 1764, and has 4 pixels per cell
classifier_Four = HogSvm_Classifier(0.5, 1764, npz_file, all_labels, posex, negex, 4)

# The four classifiers are stored in an array called classifiers
classifiers = [classifier_One, classifier_Two, classifier_Three, classifier_Four]

# Pickle is used to save (write) the train classifier in a file called linsvmhog.pkl
pickle.dump(classifiers, open("linsvmhog.pkl", "wb"))
