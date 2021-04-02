import numpy as np
import cv2
import argparse
import time

def main(): 
    first = np.load('./kernal_sum.npy')
    second = np.load('./kernal_sum{0}.npy'.format('_jin_yu'))
    print(first.shape)
    print(second.shape)
    print(first.dtype)
    print(second.dtype)
    print(first[1])
    print(second[1])
    # print(first - second)
    print(np.where(first != second))
    print(np.where(first != second)[0].shape)
    # print(sum(abs(first - second)))

def main01(): 
    parser = argparse.ArgumentParser(description='evaluation function of two joint bilateral filter result')
    parser.add_argument('--first_path', default='./testdata/2_result_accelerate/', help='path to input image')
    parser.add_argument('--second_path', default='./testdata/2_result_accelerate_ex/', help='path to ground truth bf image')
    args = parser.parse_args()

    file_nm_list = ['bf_origin.png', 'jbf_0.png', 'jbf_1.png', 'jbf_2.png', 'jbf_3.png', 'jbf_4.png', 'jbf_5.png']

    for file_nm in file_nm_list: 
        first_img = cv2.cvtColor(cv2.imread(args.first_path+file_nm), cv2.COLOR_BGR2RGB)
        second_img = cv2.cvtColor(cv2.imread(args.second_path+file_nm), cv2.COLOR_BGR2RGB)

        error = np.sum(np.abs(first_img.astype('int32')-second_img.astype('int32')))
        print('error: ', error)
        diff = np.abs(first_img.astype('int32')-second_img.astype('int32'))
        print(np.where(diff!=0))
        # print(first_img[0, 173,2])
        # print(second_img[0, 173,2])

if __name__ == "__main__": 
    main()