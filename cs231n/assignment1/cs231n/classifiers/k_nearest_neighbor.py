from builtins import range
from builtins import object
import math
import numpy as np
from past.builtins import xrange

from collections import Counter


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError("Invalid value %d for num_loops" % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0] # 500
        num_train = self.X_train.shape[0] # 5000
        dists = np.zeros((num_test, num_train))

        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################

                curr_test = X[i]
                curr_train = self.X_train[j]
                
                summed_diff = sum(pow(curr_test[k] - curr_train[k], 2) for k in range(len(curr_test)))
                dists[i][j] = math.sqrt(summed_diff)
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            curr_test = X[i]

            # axis = 1 is running horizontally across columns for each indiv row
            squared_diff = (curr_test - self.X_train) ** 2
            dists[i, :] = np.sqrt(np.sum(squared_diff, axis=1))
        return dists

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################

        '''
        initial attempt: very memory inefficient
        issue is 3D expansion takes space in memory - 7.68B floats approx

        code = """
        stretched_test = X[:, np.newaxis, :]
        stretched_train = self.X_train[np.newaxis, :, :]

        dists[:, :] = np.sqrt(((stretched_test - stretched_train) ** 2).sum(axis=-1))
        """

        stretched test becomes: (500, 1, 3072)
        stretched train becomes: (1, 5000, 3072)

        we're finally interested in 500 x 5000 matrix of distances (stretching here is to enable numpy to perform
        broadcasting like distance calculation for every test point with every train point)

        modified test becomes like: [ [ [3072 vals] ], [ [3072 vals] ], [ [3072 vals] ], ...]
        modified train becomes like: [ [5000 elements of 3072 vals each] ]

        resulting shape: (500, 5000, 3072). 1st subtraction below to begin with

        [
            [
                test0 - train0
                test0 - train1
                ...
                test0 - train5000
            ],

            ...
            [
                test500 - train0
                ...
                test500 - train500
            ]
        ]

        sum(axis=-1) is to get rid of the last dim and make it (500 x 5000 at the end)
        '''

        # gets it from (500, 3072) -> (500,)
        X_summed = np.sum((X ** 2), axis=1)

        # (5000, 3072) -> (5000,)
        X_train_summed = np.sum((self.X_train ** 2), axis=1)

        # originally would've been (500,) + (5000,) - (500, 5000)
        # but for broadcasting to work we need to reshape 1st one to attach new 2nd dimension
        squared_diff = X_summed[:, np.newaxis] + X_train_summed - (2 * (X @ self.X_train.T))
        dists[:, :] = np.sqrt(squared_diff)

        return dists

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """

        num_test = dists.shape[0] # 500
        y_pred = np.zeros(num_test) # 1D array of 500 elements (one prediction per test example)

        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = []
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################

            current_distances = dists[i]
            sorted_distances = np.argsort(current_distances)
            labels = self.y_train[sorted_distances]
            closest_y.extend(labels[:k])


            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            freq = Counter(closest_y)
            highest_label_freq = max(freq.values())

            matching_labels_with_most_freq = [label for label in freq if freq[label] == highest_label_freq]

            y_pred[i] = sorted(matching_labels_with_most_freq)[0]

        return y_pred
