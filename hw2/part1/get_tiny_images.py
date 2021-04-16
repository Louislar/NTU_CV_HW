from PIL import Image
import numpy as np
import cv2 

def get_tiny_images(image_paths):
    #############################################################################
    # TODO:                                                                     #
    # To build a tiny image feature, simply resize the original image to a very #
    # small square resolution, e.g. 16x16. You can either resize the images to  #
    # square while ignoring their aspect ratio or you can crop the center       #
    # square portion out of each image. Making the tiny images zero mean and    #
    # unit length (normalizing them) will increase performance modestly.        #
    #############################################################################
    '''
    Input : 
        image_paths: a list(N) of string where each string is an image 
        path on the filesystem.
    Output :
        tiny image features : (N, d) matrix of resized and then vectorized tiny
        images. E.g. if the images are resized to 16x16, d would equal 256.
    '''

    '''
    1. read all the imgs in image_paths 
    2. get low resolution imgs
    3. return all the low resolution imgs
    '''
    
    imgs = [cv2.imread(path) for path in image_paths]
    # resized_img = [cv2.normalize(cv2.resize(img, (16, 16)), None, 0, 1, cv2.NORM_MINMAX).flatten() for img in imgs]   # Fail
    resized_img = [cv2.resize(img, (16, 16)).flatten() for img in imgs]
    resized_img = [(i - i.mean())/(i.std()) for i in resized_img]
    tiny_images = np.array(resized_img)

    
    ##############################################################################
    #                                END OF YOUR CODE                            #
    ##############################################################################

    return tiny_images
