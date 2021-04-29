from PIL import Image
import numpy as np
from cyvlfeat.sift.dsift import dsift
from cyvlfeat.kmeans import kmeans
from time import time
import cv2

#This function will sample SIFT descriptors from the training images,
#cluster them with kmeans, and then return the cluster centers.

def build_vocabulary(image_paths, vocab_size):
    ##################################################################################
    # TODO:                                                                          #
    # Load images from the training set. To save computation time, you don't         #
    # necessarily need to sample from all images, although it would be better        #
    # to do so. You can randomly sample the descriptors from each image to save      #
    # memory and speed up the clustering. Or you can simply call vl_dsift with       #
    # a large step size here.                                                        #
    #                                                                                #
    # For each loaded image, get some SIFT features. You don't have to get as        #
    # many SIFT features as you will in get_bags_of_sift.py, because you're only     #
    # trying to get a representative sample here.                                    #
    #                                                                                #
    # Once you have tens of thousands of SIFT features from many training            #
    # images, cluster them with kmeans. The resulting centroids are now your         #
    # visual word vocabulary.                                                        #
    ##################################################################################
    ##################################################################################
    # NOTE: Some useful functions                                                    #
    # This function will sample SIFT descriptors from the training images,           #
    # cluster them with kmeans, and then return the cluster centers.                 #
    #                                                                                #
    # Function : dsift()                                                             #
    # SIFT_features is a N x 128 matrix of SIFT features                             #
    # There are step, bin size, and smoothing parameters you can                     #
    # manipulate for dsift(). We recommend debugging with the 'fast'                 #
    # parameter. This approximate version of SIFT is about 20 times faster to        #
    # compute. Also, be sure not to use the default value of step size. It will      #
    # be very slow and you'll see relatively little performance gain from            #
    # extremely dense sampling. You are welcome to use your own SIFT feature.        #
    #                                                                                #
    # Function : kmeans(X, K)                                                        #
    # X is a M x d matrix of sampled SIFT features, where M is the number of         #
    # features sampled. M should be pretty large!                                    #
    # K is the number of clusters desired (vocab_size)                               #
    # centers is a d x K matrix of cluster centroids.                                #
    #                                                                                #
    # NOTE:                                                                          #
    #   e.g. 1. dsift(img, step=[?,?], fast=True)                                    #
    #        2. kmeans( ? , vocab_size)                                              #  
    #                                                                                #
    # ################################################################################
    '''
    Input : 
        image_paths : a list of training image path
        vocal size : number of clusters desired
    Output :
        Clusters centers of Kmeans
    '''

    '''
    1. Convert to gray scale 
    2. sift 
    3. k means 
    '''

    imgs = [cv2.imread(path) for path in image_paths]

    # convert to gray image 
    gray_imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2GRAY) for i in imgs]
    print(gray_imgs[0].shape)
    print(np.array(gray_imgs).shape)

    # sift
    sifts = [dsift(i, step=[10, 10], size=3, fast=True)[1].astype(np.float64) for i in gray_imgs]
    print(sifts[0].shape)
    print(len(sifts))

    # randomly sample descriptors from sift result (concatenate them all) 
    num_of_descriptors = 250 # number of descriptors(a sift descriptors which has 128 dimensions) randomly pick from each training image 
    sifts_rand_desciptors = [mat[np.random.choice(mat.shape[0], num_of_descriptors, replace=False), :] for mat in sifts]
    print(sifts_rand_desciptors[0].shape)
    sifts_rand_desciptors = np.concatenate(sifts_rand_desciptors, axis=0)
    print(sifts_rand_desciptors.shape)

    # kmeans with k = vocab_size
    centroids = kmeans(sifts_rand_desciptors, vocab_size)
    print(centroids.shape)

    vocab = centroids


    ##################################################################################
    #                                END OF YOUR CODE                                #
    ##################################################################################
    return vocab

