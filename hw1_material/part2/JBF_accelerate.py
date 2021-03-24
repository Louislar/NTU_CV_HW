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
        gaussian_kernel = np.exp(-1*((grid_x**2 + grid_y**2) / (2*self.sigma_s**2)))
        gaussian_kernel_3d = gaussian_kernel[:, :, None]
        # print(gaussian_kernel)
        # print(gaussian_kernel.dtype)

        # 2.1 guidence image normalize [0, 1]
        normalized_padded_guidance = padded_guidance / 255.
        # 2.2 range kernel lookup table (how to do this?)
        #       Note: range kernel will only use guidance image to calculate kernel entries
        

        # 3. convolution (naiive solution)
        after_conv_mat = np.zeros((origin_img_shape_x, origin_img_shape_y, origin_img_shape_z))
        for i in range(padded_guidance.shape[0]-2*half_window_len): 
            i = i + half_window_len
            for j in range(padded_guidance.shape[1]-2*half_window_len):
                j = j + half_window_len
                window_pixels = padded_img[i-half_window_len:i+half_window_len+1, j-half_window_len:j+half_window_len+1, :]
                # print(i, ', ', j, ', ', i-half_window_len, ', ', i+half_window_len+1, ', ', j-half_window_len, ', ', j+half_window_len+1)

                if guidance_img_dim == 3: 
                    central_val = normalized_padded_guidance[i,j,:]
                    normalized_window_pixels = normalized_padded_guidance[i-half_window_len:i+half_window_len+1, j-half_window_len:j+half_window_len+1, :]
                    # calculate range kernel
                    normalized_window_pixels = np.power((normalized_window_pixels-central_val), 2)
                    sum_of_squ_of_rgb = np.sum(normalized_window_pixels, axis=2)
                    final_range_kernel = np.exp((-sum_of_squ_of_rgb)/(2*self.sigma_r**2))
                    range_kernel_3d = final_range_kernel[:, :, None]    # range kernel調整成三維
                    after_multiply_denominator = np.multiply(gaussian_kernel_3d, range_kernel_3d)


                # guidance image 是灰階圖 --> 2維陣列
                elif guidance_img_dim == 2: 
                    central_val = normalized_padded_guidance[i,j]
                    normalized_window_pixels = normalized_padded_guidance[i-half_window_len:i+half_window_len+1, j-half_window_len:j+half_window_len+1]
                    normalized_window_pixels = np.power((normalized_window_pixels - central_val), 2)
                    final_range_kernel = np.exp(-(normalized_window_pixels)/(2*self.sigma_r**2))
                    after_multiply_denominator = np.multiply(gaussian_kernel, final_range_kernel)
                    after_multiply_denominator = after_multiply_denominator[:, :, None]

                after_multiply_fraction = np.multiply(after_multiply_denominator, window_pixels)
                after_sum_denominator = np.sum(after_multiply_denominator, axis=(0, 1))
                after_sum_fraction = np.sum(after_multiply_fraction, axis=(0, 1))
                after_divide = np.divide(after_sum_fraction, after_sum_denominator)
                after_conv_mat[i-half_window_len, j-half_window_len, :] = after_divide 

        
        print(after_conv_mat.shape)
        print(after_conv_mat[0:20, 0:20])
        print(after_conv_mat.dtype)
        


        cv2.imshow('img_show', cv2.cvtColor(np.clip(after_conv_mat, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        # 按空白鍵退出
        key = None
        while True: 
            key = cv2.waitKey(0)
            if key == 32: 
                break

        cv2.destroyAllWindows()
        output = after_conv_mat
        return np.clip(output, 0, 255).astype(np.uint8)
