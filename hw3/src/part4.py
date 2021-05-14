import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping
import logging

random.seed(999)
rnd_seed = 999  # For numpy.random.seed()

def homography_and_ransac(pt_pairs1, pt_pairs2, s=8, t=1, T=10, N=10): 
    '''
    Ref: http://6.869.csail.mit.edu/fa12/lectures/lecture13ransac/lecture13ransac.pdf (page 35)
    :pt_pairs1: feature points from first image, arange by matching index (must have same size as pairs2)
    :pt_pairs2: feature points from second image, arange by matching index 
    :s: Randomly pick s points to construct H 
    :t: Distance between x_prime and H*x must smaller than t, then x will be inlier
    :T: Inlier set size bigger than T, stop re-compute H 
    :N: Maximum re-compute time 
    '''
    global rnd_seed
    bestH = None
    ptPairCount = pt_pairs1.shape[0]
    print(ptPairCount)
    iterCount = 0
    inlierIdxSetPerIter = []

    # Form points in homogeneous coordinate, and change to shape 3xN, 
    # for easy matrix multiplication 
    print('point pair 1 shape: ', pt_pairs1.shape)
    pt_pairs1_homo = np.transpose(pt_pairs1)
    pt_pairs2_homo = np.transpose(pt_pairs2)
    tmpOneArr = np.ones_like(pt_pairs1_homo[0, :])
    pt_pairs1_homo = np.vstack([pt_pairs1_homo, tmpOneArr])
    pt_pairs2_homo = np.vstack([pt_pairs2_homo, tmpOneArr])
    print('point pair 1 homogeneous shape: ', pt_pairs1_homo.shape)
    print(tmpOneArr.shape)

    # RANSAC
    while iterCount <= N:
        
        np.random.seed(rnd_seed)
        rndPickIdx = np.random.choice(ptPairCount, s)
        # print(rndPickIdx)
        rndPickPts1 = pt_pairs1[rndPickIdx, :]
        rndPickPts2 = pt_pairs2[rndPickIdx, :]
        # print(rndPickPts1.shape)
        # print(rndPickPts2.shape)
        estH = solve_homography(rndPickPts1, rndPickPts2)
        ## All the points in pair 1 needs to do transformation by homography
        afterHomography = np.dot(estH, pt_pairs1_homo)
        afterHomography[0, :] = np.divide(afterHomography[0, :], afterHomography[2, :])
        afterHomography[1, :] = np.divide(afterHomography[1, :], afterHomography[2, :])
        afterHomography[2, :] = np.divide(afterHomography[2, :], afterHomography[2, :])
        # print(afterHomography.shape)

        ## Calculate the distance between pair1 after homography and pair2, 
        ## store index which distance is lower then t (Use L2 norm as distance measurement method)
        # print(pt_pairs2_homo.shape)
        afterSubtract = np.subtract(afterHomography[:2, :], pt_pairs2_homo[:2, :])
        afterSquare = np.power(afterSubtract, 2)
        sumOfSquare = np.sum(afterSquare, axis=0)
        sumOfSquare = np.sqrt(sumOfSquare)
        # print(afterSquare.shape)
        # print(sumOfSquare.shape)
        # print(sumOfSquare)
        inlierIdxSet = np.argwhere(sumOfSquare <= t)
        inlierIdxSetPerIter.append(inlierIdxSet)
        if inlierIdxSet.shape[0] >= T: 
            pass
        print('inlier shape: ', inlierIdxSet.shape)
        print(inlierIdxSet)

        iterCount += 1
        rnd_seed += 1

    ## Use biggest inlier set to calculate homography again 
    inlierIdxSetPerIter = sorted(inlierIdxSetPerIter, key=lambda x: len(x))
    biggestInlierIdxSet = inlierIdxSetPerIter[-1][:, 0]
    print('Biggest inlier set size: ', len(biggestInlierIdxSet))
    print(biggestInlierIdxSet)
    bestH = solve_homography(pt_pairs1[biggestInlierIdxSet, :], pt_pairs2[biggestInlierIdxSet, :])


    return bestH

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]    # Assign frame 1 to destination map D
    last_best_H = np.eye(3)
    out = None

    # for all images to be stitched:
    for idx in range(len(imgs) - 1):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1.feature detection & matching
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(im1,None)  # query image
        kp2, des2 = orb.detectAndCompute(im2,None)  # train image

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)

        # Matcher object reference: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html 
        print('number of feature matched: ', len(matches))

        # Rearange matched key points
        fp1 = np.vstack([np.array(list(_kp_obj.pt)) for _kp_obj in kp1])    # feature point of first image 
        fp2 = np.vstack([np.array(list(_kp_obj.pt)) for _kp_obj in kp2])    # feature point of second image 
        # print(fp1.shape)
        # print(fp2.shape)
        matchIdx1 = [_MatcherObj.queryIdx for _MatcherObj in matches]
        matchIdx2 = [_MatcherObj.trainIdx for _MatcherObj in matches]
        # print(matchIdx1)
        # print(matchIdx2)
        matchedfp1 = fp1[matchIdx1, :]
        matchedfp2 = fp2[matchIdx2, :]
        # print(matchedfp1.shape)
        # print(matchedfp2.shape)
        # print(matchedfp1)

        # TODO: 2. apply RANSAC to choose best H
        H = homography_and_ransac(matchedfp1, matchedfp2)

        # TODO: 3. chain the homographies

        # TODO: 4. apply warping
        break
    return out


if __name__ == "__main__":

    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)