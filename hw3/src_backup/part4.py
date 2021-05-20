import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping
import logging

random.seed(999)
rnd_seed = 999  # For numpy.random.seed()

def homography_and_ransac(pt_pairs1, pt_pairs2, s=5, t=1, T=30, N=100): 
    '''
    Ref: http://6.869.csail.mit.edu/fa12/lectures/lecture13ransac/lecture13ransac.pdf (page 35)
    :pt_pairs1: feature points from first image, arange by matching index (must have same size as pairs2) (this will be destination image --> canvas)
    :pt_pairs2: feature points from second image, arange by matching index (this will be source image)
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
        # print('Random pick points 1 shape: ', rndPickPts1.shape)
        # print('Random pick points 2 shape: ', rndPickPts2.shape)
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
        # print('inlier shape: ', inlierIdxSet.shape)
        # print(inlierIdxSet)

        iterCount += 1
        rnd_seed += 1

    ## Use biggest inlier set to calculate homography again 
    inlierIdxSetPerIter = sorted(inlierIdxSetPerIter, key=lambda x: len(x))
    biggestInlierIdxSet = inlierIdxSetPerIter[-1][:, 0]
    print('Biggest inlier set size: ', len(biggestInlierIdxSet))
    # print(biggestInlierIdxSet)
    ### If all the inlier set contains nothing (size == 0), then break 
    if len(biggestInlierIdxSet) <= 0: 
        print('No homography found')
        return None
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

    dst_list = [dst.copy(), np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8), np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)]

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
        H = homography_and_ransac(matchedfp2, matchedfp1)   # img2 translate to img1 --> img1 coord = H * im2 coord 

        # TODO: 3. chain the homographies
        last_best_H = np.dot(last_best_H, H)

        # TODO: 4. apply warping
        ## Question: What is the destination coordinate? --> (ymin, ymax, xmin, xmax)?
        # print(imgs[0].shape[0])
        # print(imgs[0].shape[1])
        # print(imgs[1].shape[0])
        # print(imgs[1].shape[1])
        # print(sum([i.shape[0] for i in imgs[:idx+1]]))
        # print(sum([i.shape[0] for i in imgs[:idx+2]]))
        # print(sum([i.shape[1] for i in imgs[:idx+1]]))
        # print(sum([i.shape[1] for i in imgs[:idx+2]]))
        dst = warping(im2, dst, last_best_H, 
            0, 
            imgs[0].shape[0], 
            sum([i.shape[1] for i in imgs[:idx+1]]), 
            sum([i.shape[1] for i in imgs[:idx+2]]), 
            direction='b')



        # TODO: 5. alpha blending
        dst_list[idx+1] = warping(im2, dst_list[idx+1], last_best_H, 
            0, 
            dst.shape[0], 
            0, 
            dst.shape[1], 
            direction='b')
        # break
    dst = alpha_blending(dst_list)
    out = dst
    return out

def alpha_blending(CanvasImgList): 
    '''
    Three same shape image(canvas), which has paste different source image on the cancas 
    Just apply simple alpha blending, which means that alpha = 0.5
    Ref: https://inst.eecs.berkeley.edu/~cs194-26/fa17/upload/files/proj6B/cs194-26-abw/ 
    '''
    newCanvas = np.zeros((CanvasImgList[0].shape[0], CanvasImgList[0].shape[1], CanvasImgList[0].shape[2]), dtype=np.uint8)
    for idx in range(len(CanvasImgList)):
        im1 = CanvasImgList[idx]


        # Construct mask (Two images in a pair)
        mask1 = np.where(im1>0, 1, 0)
        mask2 = np.where(newCanvas>0, 2, 0)
        mask = mask1 + mask2
        
        
        newCanvas[mask==1] = im1[mask==1] * 1.0 + newCanvas[mask==1] * 0.0
        newCanvas[mask==2] = im1[mask==2] * 0.0 + newCanvas[mask==2] * 1.0
        newCanvas[mask==3] = im1[mask==3] * 0.5 + newCanvas[mask==3] * 0.5

    # TODO: Old version with wrong result
    # for idx in range(len(CanvasImgList) - 1):
    #     im1 = CanvasImgList[idx]
    #     im2 = CanvasImgList[idx + 1]


    #     # Construct mask (Two images in a pair)
    #     mask1 = np.where(im1>0, 1, 0)
    #     mask2 = np.where(im2>0, 2, 0)
    #     mask = mask1 + mask2
        
        
    #     newCanvas[mask==1] = im1[mask==1] * 1.0 + im2[mask==1] * 0.0
    #     newCanvas[mask==2] = im1[mask==2] * 0.0 + im2[mask==2] * 1.0
    #     newCanvas[mask==3] = im1[mask==3] * 0.5 + im2[mask==3] * 0.5
    
    # cv2.imwrite('output_test.png', newCanvas)
    return newCanvas

if __name__ == "__main__":

    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)