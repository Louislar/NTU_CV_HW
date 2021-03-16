import numpy as np
import cv2
import matplotlib.pyplot as plt


class Harris_corner_detector(object):
    def __init__(self, threshold):
        self.threshold = threshold

    def detect_harris_corners(self, img):
        ### TODO ####
        # Step 1: Smooth the image by Gaussian kernel
        # - Function: cv2.GaussianBlur (kernel = 3, sigma = 1.5)
        smooth_image = cv2.GaussianBlur(img, (3, 3), 1.5)

        # Step 2: Calculate Ix, Iy (1st derivative of image along x and y axis)
        # - Function: cv2.filter2D (kernel = [[1.,0.,-1.]] for Ix or [[1.],[0.],[-1.]] for Iy)
        kernel_for_Ix = np.array([
            [0., 0., 0.], 
            [1.,0.,-1.], 
            [0., 0., 0.]
        ])
        kernel_for_Iy = np.array([
            [0., 1., 0.],
            [0., 0., 0.],
            [0., -1., 0.]
        ])
        Ix = cv2.filter2D(smooth_image, -1, kernel_for_Ix)
        Iy = cv2.filter2D(smooth_image, -1, kernel_for_Iy)
        print(Ix.dtype)

        # Step 3: Compute Ixx, Ixy, Iyy (Ixx = Ix*Ix, ...)
        Ixx = np.multiply(Ix, Ix) # 這邊是dot product還是Hadamard product?
        Ixy = np.multiply(Ix, Iy)
        Iyy = np.multiply(Iy, Iy)
        print(Ixx.dtype)

        # Step 4: Compute Sxx, Sxy, Syy (weighted summation of Ixx, Ixy, Iyy in neighbor pixels)
        # - Function: cv2.GaussianBlur (kernel = 3, sigma = 1.)
        Sxx = cv2.GaussianBlur(Ixx, (3, 3), 1.)
        Sxy = cv2.GaussianBlur(Ixy, (3, 3), 1.)
        Syy = cv2.GaussianBlur(Iyy, (3, 3), 1.)
        print(Sxx.dtype)

        # Step 5: Compute the det and trace of matrix M (M = [[Sxx, Sxy], [Sxy, Syy]])
        det = np.zeros(Sxx.shape)
        trace = np.zeros(Sxx.shape)
        for i in range(Sxx.shape[0]): 
            for j in range(Sxx.shape[1]): 
                M = np.array([[Sxx[i, j], Sxy[i, j]], [Sxy[i, j], Syy[i, j]]])
                det[i, j] = np.linalg.det(M)
                trace[i, j] = np.trace(M)
        print(det.dtype)
        print(trace.dtype)

        # Step 6: Compute the response of the detector by det/(trace+1e-12)
        R_matrix = np.zeros(Sxx.shape)
        for i in range(Sxx.shape[0]): 
            for j in range(Sxx.shape[1]): 
                R_matrix[i, j] = det[i, j]/(trace[i, j]+10**-12)
        print(R_matrix.dtype)

        # print(R_matrix.shape) # (256, 256)

        # return 0
        return R_matrix
    
    def post_processing(self, response):
        ### TODO ###
        # Step 1: Thresholding
        retval, after_thresholding = cv2.threshold(response, self.threshold, 255., cv2.THRESH_BINARY)
        candidate_pt_list = list(zip(*np.where(after_thresholding==255.)))
        # Step 2: Find local maximum
        local_maximum_list = []
        after_padding = cv2.copyMakeBorder(response, 2, 2, 2, 2, cv2.BORDER_CONSTANT, 0)
        for i, j in candidate_pt_list: 
            i = i+2
            j = j+2
            tmp_mat = after_padding[i-2:i+3, j-2:j+3]
            central_val = after_padding[i, j]
            maximum_value = np.amax(tmp_mat)
            # print(np.isin(maximum_value, np.unique(tmp_mat)))
            if central_val >= maximum_value: 
                if np.isin(maximum_value, np.unique(tmp_mat)): 
                    local_maximum_list.append([i-2, j-2])

        # print(local_maximum_list)
        cv2.imshow('img_show', after_padding)
        # 按空白鍵退出
        key = None
        while True: 
            key = cv2.waitKey(0)
            if key == 32: 
                break

        cv2.destroyAllWindows()

        # return 0
        return local_maximum_list