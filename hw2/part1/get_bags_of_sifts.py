from PIL import Image
import numpy as np
from scipy.spatial import distance
import pickle
import scipy.spatial.distance as distance
from cyvlfeat.sift.dsift import dsift
from time import time
import cv2

def get_bags_of_sifts(image_paths):
    ############################################################################
    # TODO:                                                                    #
    # This function assumes that 'vocab.pkl' exists and contains an N x 128    #
    # matrix 'vocab' where each row is a kmeans centroid or visual word. This  #
    # matrix is saved to disk rather than passed in a parameter to avoid       #
    # recomputing the vocabulary every time at significant expense.            #
    #                                                                          #                                                               
    # image_feats is an N x d matrix, where d is the dimensionality of the     #
    # feature representation. In this case, d will equal the number of clusters#
    # or equivalently the number of entries in each image's histogram.         #
    #                                                                          #
    # You will construct SIFT features here in the same way you did in         #
    # build_vocabulary (except for possibly changing the sampling rate)        #
    # and then assign each local feature to its nearest cluster center         #
    # and build a histogram indicating how many times each cluster was used.   #
    # Don't forget to normalize the histogram, or else a larger image with more#
    # SIFT features will look very different from a smaller version of the same#
    # image.                                                                   #
    ############################################################################
    '''
    Input : 
        image_paths : a list(N) of training images
    Output : 
        image_feats : (N, d) feature, each row represent a feature of an image
                    N: number of images (input images) (training images)
                    d: number of clusters (number of kmeans centroids) (vocabluary size)
    '''
    print('\n======= Get bag of sifts =======')

    imgs = [cv2.imread(path) for path in image_paths]
    gray_imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in imgs]
    # sift
    sifts = [dsift(i, step=[3, 3], size=5, fast=True)[1].astype(np.float64) for i in gray_imgs]
    print('Training images and sift\'s shape: ', (len(sifts), sifts[0].shape))
    
    # TODO
    # randomly sample descriptors from sift result (concatenate them all)
    num_of_descriptors = 500
    np.random.seed(0)
    sifts_rand_descriptors = [mat[np.random.choice(mat.shape[0], num_of_descriptors, replace=False), :] for mat in sifts]
    print('Randomly pick sift descriptor\'s shape: ', len(sifts_rand_descriptors), sifts_rand_descriptors[0].shape)

    # read vocab.pkl 
    vocab = None 
    with open('vocab.pkl', 'rb') as handle: 
        vocab = pickle.load(handle)
    print('Vocabulary matrix\'s shape: ', vocab.shape)

    # calculate each input image's histogram (also normalize it) 
    # l2 norm of each feaure and vocabulary
    vocab_dist = [distance.cdist(train_feat, vocab) for train_feat in sifts]
    print('Vocabulary distance matrix\'s shape: ', (len(vocab_dist), vocab_dist[0].shape))
    # nearest vocabulary of each feature in each training image 
    nearest_vocab = [np.argsort(dist, axis=1)[:, 0] for dist in vocab_dist]
    print('Nearest vocabulary matrix\'s shape: ', (len(nearest_vocab), nearest_vocab[0].shape))
    # construct each training image's histogram 
    vocab_hist = [np.bincount(c_vocab, minlength=vocab.shape[0]) for c_vocab in nearest_vocab]
    normalize_vocab_hist = [h_vocab / np.max(h_vocab) for h_vocab in vocab_hist]
    print('Vocabulary histogram matrix\'s shape: ', (len(vocab_hist), vocab_hist[0].shape))
    # make into numpy array with shape = (N, d) 
    image_feats = np.vstack(normalize_vocab_hist)
    # image_feats = np.concatenate(normalize_vocab_hist, axis=0)
    print('Vocabulary histogram matrix\'s shape: ', image_feats.shape)



    
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return image_feats
