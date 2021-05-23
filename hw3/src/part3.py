import numpy as np
import cv2
from utils import solve_homography, warping


if __name__ == '__main__':

    # ================== Part 3 ========================
    secret1 = cv2.imread('../resource/BL_secret1.png')
    secret2 = cv2.imread('../resource/BL_secret2.png')
    corners1 = np.array([[429, 337], [517, 314], [570, 361], [488, 380]])
    corners2 = np.array([[346, 196], [437, 161], [483, 198], [397, 229]])
    h, w, c = (500, 500, 3)
    dst1 = np.zeros((h, w, c))
    dst2 = np.zeros((h, w, c))

    # TODO: call solve_homography() & warping
    H1 = solve_homography(corners1, np.array([[0, 0], [w, 0], [w, h], [0, h]]))
    H2 = solve_homography(corners2, np.array([[0, 0], [w, 0], [w, h], [0, h]]))


    output3_1 = warping(secret1, dst1, H1, 0, h, 0, w, 'b')
    output3_2 = warping(secret2, dst2, H2, 0, h, 0, w, 'b')

    cv2.imwrite('output3_1.png', output3_1)
    cv2.imwrite('output3_2.png', output3_2)
    cv2.imwrite('output3_1_1.png', dst1)
    cv2.imwrite('output3_2_1.png', dst2)

    dstFull1 = np.zeros(secret1.shape)
    dstFull2 = np.zeros(secret1.shape)
    output3_3 = warping(secret1, dstFull1, H1, 0, dstFull1.shape[0], 0, dstFull1.shape[1], 'b')
    output3_4 = warping(secret2, dstFull2, H2, 0, dstFull1.shape[0], 0, dstFull1.shape[1], 'b')
    cv2.imwrite('output3_3.png', output3_3)
    cv2.imwrite('output3_4.png', output3_4)
    print(output3_3.shape)
    print(output3_4.shape)