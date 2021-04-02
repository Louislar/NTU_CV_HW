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
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

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
        gaussian_kernel = np.exp(-1*((grid_x**2 + grid_y**2) / (2*self.sigma_s**2))).flatten()
        gaussian_kernel = np.repeat(gaussian_kernel[:, None], img.shape[0]*img.shape[1], axis=1)
        gaussian_kernel = np.transpose(gaussian_kernel)
        # np.save('gaussian_kernel', gaussian_kernel)
        # print(gaussian_kernel)
        # print(gaussian_kernel.shape)

        # 2.1 guidence image normalize [0, 1]
        normalized_padded_guidance = padded_guidance / 255.
        # 2.2 range kernel lookup table (how to do this?)
        #       Note: range kernel will only use guidance image to calculate kernel entries
        unique_value = np.unique(padded_guidance)
        unique_value_normalized = np.unique(normalized_padded_guidance)
        new_unique_value = np.zeros(256)
        new_unique_value[unique_value] = unique_value_normalized
        unique_value = new_unique_value
        lookup_table_mat = unique_value[:, None]-unique_value
        
        lookup_table_mat = np.exp(-(lookup_table_mat**2)/(2*self.sigma_r**2))
        # np.save('lookup_table', lookup_table_mat)
        # print(self.sigma_r)
        

        # print(lookup_table_mat.shape) 
        

        # 3. unroll convolution to inner product
        def unroll_conv(img, height, width, k_height, k_width, move_height, move_width): 
            grid_x, grid_y = np.meshgrid(
                        np.arange(k_width), 
                        np.arange(k_height)
                    )
            idx_mat = list(zip(grid_y.flatten(), grid_x.flatten()))
            unrolled_mat = np.zeros(( k_width*k_height, move_width*move_height)).astype(np.int32)
            for row_i, col_i in idx_mat: 
                unrolled_mat[row_i*k_width+col_i, :] = img[row_i:(row_i+move_height), col_i:(col_i+move_width)].flatten()
            unroll_trans = np.transpose(unrolled_mat)
            return unroll_trans

        

        # 3.1 cal range kernel
        range_kernel = None
        if guidance_img_dim ==3: 
            unrolled_guidence_0 = unroll_conv(
                padded_guidance[:, :, 0], 
                padded_guidance.shape[0], padded_guidance.shape[1], 
                self.wndw_size, self.wndw_size, 
                guidance.shape[0], guidance.shape[1]
            )
            unrolled_guidence_1 = unroll_conv(
                padded_guidance[:, :, 1], 
                padded_guidance.shape[0], padded_guidance.shape[1], 
                self.wndw_size, self.wndw_size, 
                guidance.shape[0], guidance.shape[1]
            )
            unrolled_guidence_2 = unroll_conv(
                padded_guidance[:, :, 2], 
                padded_guidance.shape[0], padded_guidance.shape[1], 
                self.wndw_size, self.wndw_size, 
                guidance.shape[0], guidance.shape[1]
            )
            unrolled_guidence_mid_0 = unrolled_guidence_0[:, (unrolled_guidence_0.shape[1]//2)]
            unrolled_guidence_mid_1 = unrolled_guidence_1[:, (unrolled_guidence_1.shape[1]//2)]
            unrolled_guidence_mid_2 = unrolled_guidence_2[:, (unrolled_guidence_2.shape[1]//2)]
            range_kernel_0 = lookup_table_mat[unrolled_guidence_0, unrolled_guidence_mid_0[:, None]]
            range_kernel_1 = lookup_table_mat[unrolled_guidence_1, unrolled_guidence_mid_1[:, None]]
            range_kernel_2 = lookup_table_mat[unrolled_guidence_2, unrolled_guidence_mid_2[:, None]]
            range_kernel = np.multiply(np.multiply(range_kernel_0, range_kernel_1), range_kernel_2)
        elif guidance_img_dim == 2: 
            unrolled_guidence = unroll_conv(
                padded_guidance, 
                padded_guidance.shape[0], padded_guidance.shape[1], 
                self.wndw_size, self.wndw_size, 
                guidance.shape[0], guidance.shape[1]
            )
            unrolled_guidence_mid = unrolled_guidence[:, (unrolled_guidence.shape[1]//2)]
            range_kernel = lookup_table_mat[unrolled_guidence, unrolled_guidence_mid[:, None]]
        
        # print(range_kernel[:10, :10])
        # np.save('range_kernel', range_kernel)
        combined_kernel = np.multiply(range_kernel, gaussian_kernel)
        sum_of_combined_kernel = np.sum(combined_kernel, axis=1)

        # print(sum_of_combined_kernel.shape)

        unrolled_img_0 = unroll_conv(
            padded_img[:, :, 0], 
            padded_img.shape[0], padded_img.shape[1], 
            self.wndw_size, self.wndw_size, 
            img.shape[0], img.shape[1]
        )
        unrolled_img_1 = unroll_conv(
            padded_img[:, :, 1], 
            padded_img.shape[0], padded_img.shape[1], 
            self.wndw_size, self.wndw_size, 
            img.shape[0], img.shape[1]
        )
        unrolled_img_2 = unroll_conv(
            padded_img[:, :, 2], 
            padded_img.shape[0], padded_img.shape[1], 
            self.wndw_size, self.wndw_size, 
            img.shape[0], img.shape[1]
        )
        result_0 = np.multiply(unrolled_img_0, combined_kernel)
        result_0 = np.sum(result_0, axis=1)
        result_0 = np.divide(result_0, sum_of_combined_kernel)
        result_0 = result_0.reshape((img.shape[0], img.shape[1]))

        result_1 = np.multiply(unrolled_img_1, combined_kernel)
        result_1 = np.sum(result_1, axis=1)
        result_1 = np.divide(result_1, sum_of_combined_kernel)
        result_1 = result_1.reshape((img.shape[0], img.shape[1]))

        result_2 = np.multiply(unrolled_img_2, combined_kernel)
        result_2 = np.sum(result_2, axis=1)
        result_2 = np.divide(result_2, sum_of_combined_kernel)
        result_2 = result_2.reshape((img.shape[0], img.shape[1]))
        # print(result_0.shape)
        # print(result_1.shape)
        # print(result_2.shape)
        after_conv_mat = np.repeat( result_0[:, :, None], 3, axis=2)
        after_conv_mat[:, :, 1] = result_1
        after_conv_mat[:, :, 2] = result_2
        


        
        # print(after_conv_mat.shape)
        # print(after_conv_mat[0:20, 0:20])
        # print(after_conv_mat.dtype)
        


        # cv2.imshow('img_show', cv2.cvtColor(np.clip(after_conv_mat, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        # # 按空白鍵退出
        # key = None
        # while True: 
        #     key = cv2.waitKey(0)
        #     if key == 32: 
        #         break

        # cv2.destroyAllWindows()
        output = after_conv_mat
        return np.clip(output, 0, 255).astype(np.uint8)
