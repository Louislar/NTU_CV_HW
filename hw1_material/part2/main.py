import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter
from collections import Counter

def read_arg_file(txt_path):
    rgb_weight_list = []
    sigma_s = None
    sigma_r = None

    with open(txt_path, 'r') as open_file: 
        lines_list = open_file.readlines()
        lines_list = [i.replace('\n', '').split(',') for i in lines_list]
        rgb_weight_list = [np.array([float(j) for j in i]) for i in lines_list[1:6]]
        sigma_s = int(lines_list[6][1])
        sigma_r = float(lines_list[6][3])
        
    print(rgb_weight_list)
    print('sigma_s: ', sigma_s)
    print('sigma_r: ', sigma_r)
    return rgb_weight_list, sigma_s, sigma_r

def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()
    result_dir = args.image_path.replace('.png', '') + '_result/'
    rgb_weight_list, sigma_s, sigma_r = read_arg_file(args.setting_path)

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ### TODO ###
    # 1. gray scale image (6 of them) (cv2的算好了)
    print(img_rgb.shape)
    gray_scale_imgs = [img_gray]
    for rgb_weight in rgb_weight_list: 
        after_multiply = np.multiply(img, rgb_weight)
        after_sum = np.sum(after_multiply, axis=2)
        after_sum = after_sum.astype('uint8')
        gray_scale_imgs.append(after_sum)

    # 2.0 bilater filter of origin img 
    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    origin_bf = JBF.joint_bilateral_filter(img_rgb, img_rgb).astype(np.uint8)
    # 2. joint bilater with gray img as guidence
    after_jbf_list = []
    for a_gray_img in gray_scale_imgs: 
        jbf_out = JBF.joint_bilateral_filter(img_rgb, a_gray_img).astype(np.uint8)
        after_jbf_list.append(jbf_out)
    
    # 3. l1 norm between each jbf_out and origin gray scale
    l1_norm_list = []
    for a_jbf_out in after_jbf_list: 
        l1_norm_list.append(
            np.sum(np.abs(np.subtract(a_jbf_out, origin_bf)))
        )
    print('lowest cost idx: ', np.argmin(l1_norm_list))

    # 4. save img/l1 cost
    # 4.1 save gray img
    for i in range(len(gray_scale_imgs)): 
        cv2.imwrite(result_dir+'gray_{0}.png'.format(i), gray_scale_imgs[i])
    # 4.2 save after jbf img
    origin_bf = cv2.cvtColor(origin_bf,cv2.COLOR_RGB2BGR)
    cv2.imwrite(result_dir+'bf_origin.png', origin_bf)
    for i in range(len(after_jbf_list)): 
        after_jbf_list[i] = cv2.cvtColor(after_jbf_list[i],cv2.COLOR_RGB2BGR)
        cv2.imwrite(result_dir+'jbf_{0}.png'.format(i), after_jbf_list[i])
    # 4.3 save l1 norm
    with open(result_dir+'l1norm.txt', 'w+') as out_f: 
        for a_l1 in [str(i) for i in l1_norm_list]: 
            out_f.writelines(a_l1 + '\n')


    # cv2.imshow('img_show', gray_scale_imgs[2])
    # # 按空白鍵退出
    # key = None
    # while True: 
    #     key = cv2.waitKey(0)
    #     if key == 32: 
    #         break
    # cv2.destroyAllWindows()

    




if __name__ == '__main__':
    main()