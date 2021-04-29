from __future__ import print_function

import numpy as np
import scipy.spatial.distance as distance
from scipy.stats import mode

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
    ###########################################################################
    # TODO:                                                                   #
    # This function will predict the category for every test image by finding #
    # the training image with most similar features. Instead of 1 nearest     #
    # neighbor, you can vote based on k nearest neighbors which will increase #
    # performance (although you need to pick a reasonable value for k).       #
    ###########################################################################
    ###########################################################################
    # NOTE: Some useful functions                                             #
    # distance.cdist :                                                        #
    #   This function will calculate the distance between two list of features#
    #       e.g. distance.cdist(? ?)                                          #
    ###########################################################################
    '''
    Input : 
        train_image_feats : 
            image_feats is an (N, d) matrix, where d is the 
            dimensionality of the feature representation.

        train_labels : 
            image_feats is a list of string, each string
            indicate the ground truth category for each training image. 

        test_image_feats : 
            image_feats is an (M, d) matrix, where d is the 
            dimensionality of the feature representation.
    Output :
        test_predicts : 
            a list(M) of string, each string indicate the predict
            category for each testing image.
    '''
    distance_mat = distance.cdist(train_image_feats, test_image_feats)
    train_labels = np.array(train_labels)

    # k=1
    # nearest_idx = np.argmin(distance_mat, axis=0)
    # test_predicts = train_labels[nearest_idx]   # k=1

    # k=x
    k = 250
    nearest_idx = np.argsort(distance_mat, axis=0)
    nearest_idx = nearest_idx[:k, :]
    # nearest_idx = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=nearest_idx)
    def _func(_arr): 
        _bincount = np.bincount(_arr)
        highest_count = np.max(_bincount)
        highest_count_idx = np.where(_bincount==highest_count)[0]
        first_highest_count = _arr[np.isin(_arr, highest_count_idx)][0]
        return first_highest_count
    nearest_idx = np.apply_along_axis(lambda x: _func(x), axis=0, arr=nearest_idx)
    test_predicts = train_labels[nearest_idx]
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return test_predicts
