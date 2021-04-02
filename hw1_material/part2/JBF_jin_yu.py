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

        # To do
        imgrow = img.shape[0]
        imgcolumn = img.shape[1]

        oimgr = padded_img[:,:,0]
        oimgg = padded_img[:,:,1]
        oimgb = padded_img[:,:,2]

        # original image Iq
        pd_original_r = np.zeros(( self.wndw_size**2,imgrow*imgcolumn),dtype=np.uint8)
        pd_original_g = np.zeros(( self.wndw_size**2,imgrow*imgcolumn),dtype=np.uint8)
        pd_original_b = np.zeros(( self.wndw_size**2,imgrow*imgcolumn),dtype=np.uint8)
        k = 0
        for i in range(self.wndw_size):
            for j in range(self.wndw_size): 
                pd_original_r[k,:] = oimgr[i:i+imgrow,j:j+imgcolumn].flatten()
                pd_original_g[k,:] = oimgg[i:i+imgrow,j:j+imgcolumn].flatten()
                pd_original_b[k,:] = oimgb[i:i+imgrow,j:j+imgcolumn].flatten()
                k = k+1

        Iq_r = np.transpose(pd_original_r)
        Iq_g = np.transpose(pd_original_g)
        Iq_b = np.transpose(pd_original_b)
        

        ''' spacial kernal'''
        # spacial kernal 
        spacial_kernal = np.zeros((imgrow*imgcolumn,self.wndw_size**2))
        imtersp = []
        for i in range(self.wndw_size):
            for j in range(self.wndw_size):
                a = (self.wndw_size//2 - i)**2 + (self.wndw_size//2 -j)**2
                b = 2*(self.sigma_s**2 )
                num_ = np.exp(-a/b)
                imtersp.append(num_)
        
        
        spacial_kernal = np.tile(imtersp,(imgrow*imgcolumn,1))


        '''range kernal'''
        # range kernal
        dim = padded_guidance.ndim
        # build look up table
        lookuptable = np.zeros((256,256), dtype=np.float64)
        for i in range(256):
            for j in range(256):
                pixel_diff = (i/255.-j/255.)
                if i == 1 & j==1: 
                    # print(type(pixel_diff))
                    # print(type(i))
                    pass
                lookuptable[i,j] =  np.exp(-((pixel_diff**2) / (2* (self.sigma_r**2))))
                # lookuptable[i,j] = pixel_diff
        print(type(self.sigma_r))
        
        range_kernal = None
        if dim == 3:
            guidance_r = padded_guidance[:,:,0]
            guidance_g = padded_guidance[:,:,1]
            guidance_b = padded_guidance[:,:,2]

            kernal_r = np.zeros(( self.wndw_size**2 , imgrow*imgcolumn ), dtype=np.int64)
            kernal_g = np.zeros(( self.wndw_size**2 , imgrow*imgcolumn ), dtype=np.int64)
            kernal_b = np.zeros(( self.wndw_size**2 , imgrow*imgcolumn ), dtype=np.int64)
            range_kernal = []

            k = 0
            for i in range(self.wndw_size):
                for j in range(self.wndw_size):
                    kernal_r[k,:] = guidance_r[i:i+imgrow,j:j+imgcolumn].flatten()
                    kernal_g[k,:] = guidance_g[i:i+imgrow,j:j+imgcolumn].flatten()
                    kernal_b[k,:] = guidance_b[i:i+imgrow,j:j+imgcolumn].flatten()
                    k += 1
            
            kernal_rt = np.transpose(kernal_r)
            kernal_gt = np.transpose(kernal_g)
            kernal_bt = np.transpose(kernal_b)

            centervalue_rt = kernal_rt[:,self.wndw_size**2 //2]
            centervalue_gt = kernal_gt[:,self.wndw_size**2 //2]
            centervalue_bt = kernal_bt[:,self.wndw_size**2 //2]

            range_kernal = lookuptable[kernal_rt,centervalue_rt[:,None]] * lookuptable[kernal_gt,centervalue_gt[:,None]] *lookuptable[kernal_bt,centervalue_bt[:,None]]

        else:
            single_rkernal = np.zeros(( self.wndw_size**2 , imgrow*imgcolumn ), dtype=np.int64)
            range_kernal = []
            k = 0
            for i in range(self.wndw_size):
                for j in range(self.wndw_size):
                    single_rkernal[k,:] = padded_guidance[i:i+imgrow,j:j+imgcolumn].flatten()
                    k += 1

            srkernal = np.transpose(single_rkernal)
            
            centervalue = srkernal[:,self.wndw_size**2 //2]
            range_kernal = lookuptable[srkernal,centervalue[:,None]]

        # print(range_kernal[:10, :10])
        np.save('lookup_table_jin_yu', lookuptable)
        np.save('range_kernel_jin_yu', range_kernal)
        np.save('gaussian_kernel_jin_yu', spacial_kernal)

        #kernal運算

        kernal = np.multiply(spacial_kernal, range_kernal)
        kernal_sum = np.sum(kernal, axis=1)
        np.save('kernal_sum_jin_yu', kernal_sum)

        img_mergedr = np.sum(kernal*Iq_r, axis = 1)
        img_mergedg = np.sum(kernal*Iq_g, axis = 1)
        img_mergedb = np.sum(kernal*Iq_b, axis = 1)

        img_rs = np.reshape(img_mergedr / kernal_sum,(imgrow,imgcolumn))
        img_gs = np.reshape(img_mergedg / kernal_sum,(imgrow,imgcolumn))
        img_bs = np.reshape(img_mergedb / kernal_sum,(imgrow,imgcolumn))

        
        output = np.dstack((img_rs,img_gs,img_bs))
        return np.clip(output, 0, 255).astype(np.uint8)
        