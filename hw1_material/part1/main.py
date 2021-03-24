import numpy as np
import cv2
import argparse
from HCD import Harris_corner_detector


def main():
    parser = argparse.ArgumentParser(description='main function of Harris corner detector')
    parser.add_argument('--threshold', default=100., type=float, help='threshold value to determine corner')
    parser.add_argument('--image_path', default='./testdata/ex.png', help='path to input image')
    args = parser.parse_args()

    print('Processing %s ...'%args.image_path)
    img = cv2.imread(args.image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)

    ### TODO ###
    HCD = Harris_corner_detector(args.threshold)
    
    response = HCD.detect_harris_corners(img_gray)
    result = HCD.post_processing(response)
    print(result)

    # cv2.circle(img, (2, 252), 10, (0, 0, 255), -1)
    for [x, y] in result: 
        cv2.circle(img, (y, x), 2, (0, 0, 255), -1)


    # 儲存image
    cv2.imwrite('./testdata/threshold_100_2.png', img)
    
    
    # 顯示image
    # cv2.imshow('img_show', img)
    # # 按空白鍵退出
    # key = None
    # while True: 
    #     key = cv2.waitKey(0)
    #     if key == 32: 
    #         break
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()