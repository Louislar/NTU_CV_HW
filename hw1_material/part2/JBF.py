import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE)

        ### TODO ###
        origin_img_shape_x = img.shape[0]   # lena: 316*316
        origin_img_shape_y = img.shape[1]
        origin_img_shape_z = img.shape[2]
        half_window_len = self.pad_w
        guidance_img_dim = padded_guidance.ndim

        # 1. gaussian kernel lookup table 
        
        grid_x, grid_y = np.meshgrid(
            np.linspace(-(half_window_len), half_window_len, self.wndw_size, dtype=np.float64), 
            np.linspace(-(half_window_len), half_window_len, self.wndw_size, dtype=np.float64)
        )
        gaussian_kernel = np.exp(-1*(grid_x**2 + grid_y**2 / (2*self.sigma_s**2)))
        gaussian_kernel_3d = np.repeat(gaussian_kernel[:, :, np.newaxis], 3, axis=2)
        # print(gaussian_kernel)
        # print(gaussian_kernel.dtype)

        # 2.1 guidence image normalize [0, 1]
        normalized_padded_guidance = padded_guidance / 255.
        # 2.2 range kernel lookup table
        

        # 3. convolution (naiive solution)
        after_conv_mat = np.zeros((origin_img_shape_x, origin_img_shape_y, origin_img_shape_z))
        for i in range(padded_guidance.shape[0]-2*half_window_len): 
            i = i + half_window_len
            for j in range(padded_guidance.shape[1]-2*half_window_len):
                j = j + half_window_len
                if guidance_img_dim == 3: 
                    central_val = normalized_padded_guidance[i,j,:]
                    normalized_window_pixels = normalized_padded_guidance[i-half_window_len:i+half_window_len+1, j-half_window_len:j+half_window_len+1, :]
                    window_pixels = padded_guidance[i-half_window_len:i+half_window_len+1, j-half_window_len:j+half_window_len+1, :]
                    # print(i, ', ', j, ', ', i-half_window_len, ', ', i+half_window_len+1, ', ', j-half_window_len, ', ', j+half_window_len+1)
                    for k in range(3): 
                        normalized_window_pixels[:,:,k] = (normalized_window_pixels[:,:,k]-central_val[k])**2

                    sum_of_rgb = normalized_window_pixels[:,:,0] + normalized_window_pixels[:,:,1] + normalized_window_pixels[:,:,2]
                    final_range_kernel = np.exp((-sum_of_rgb)/(2*self.sigma_r))
                    # print(final_range_kernel.shape)

                    # spatial, range kernel調整成三維
                    # ref: https://stackoverflow.com/questions/32171917/copy-2d-array-into-3rd-dimension-n-times-python
                    range_kernel_3d = np.repeat(final_range_kernel[:, :, np.newaxis], 3, axis=2)
                    # print(range_kernel_3d.shape)
                    
                    # 三個矩陣element wise相乘，最後加總 
                    # TODO: 感覺不大對，range kernel已經將rgb合併在一起，現在又全部展開成3維 (range kernel在算的是兩個pixel的距離)
                    after_multiply_fraction = np.multiply(np.multiply(gaussian_kernel_3d, range_kernel_3d), window_pixels)
                    after_multiply_denominator = np.multiply(gaussian_kernel_3d, range_kernel_3d)
                    after_divide = np.divide(after_multiply_fraction, after_multiply_denominator)
                    
                    # print(after_multiply.shape)

                    for k in range(3): 
                        after_conv_mat[i-half_window_len, j-half_window_len, k] = \
                            after_divide[k].sum()
        
        print(after_conv_mat.shape)
        


        cv2.imshow('img_show', after_conv_mat)
        # 按空白鍵退出
        key = None
        while True: 
            key = cv2.waitKey(0)
            if key == 32: 
                break

        cv2.destroyAllWindows()
        output = after_conv_mat
        return np.clip(output, 0, 255).astype(np.uint8)
