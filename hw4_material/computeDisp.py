import numpy as np
import cv2.ximgproc as xip
import cv2
# import 

def lbpCalculate(imgIn, h, w): 
    '''
    :imgIn: gray scale image
    :return: lbp array (h, w, 8)
    '''
    imgInPad = cv2.copyMakeBorder(imgIn, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 256)
    row_list = []
    for i in range(1,h+1): 
        _lbp_list = []
        for j in range(1,w+1): 
            _lbp = np.array([
                imgInPad[i, j] > imgInPad[i-1, j-1], 
                imgInPad[i, j] > imgInPad[i-1, j], 
                imgInPad[i, j] > imgInPad[i-1, j+1], 
                imgInPad[i, j] > imgInPad[i, j-1], 
                imgInPad[i, j] > imgInPad[i, j+1], 
                imgInPad[i, j] > imgInPad[i+1, j-1], 
                imgInPad[i, j] > imgInPad[i+1, j], 
                imgInPad[i, j] > imgInPad[i+1, j+1]
            ])
            _lbp_list.append(_lbp)
        _a_row_arr = np.vstack(_lbp_list)
        row_list.append(_a_row_arr)
    imgInLbp = np.vstack(row_list)
    imgInLbp = imgInLbp.reshape((h, w, 8))
    return imgInLbp

def computeCensusCost(Il_lbp, Ir_lbp, max_disp, shift_direction='r'): 
    '''
    Compute single channel imgaes census cost of each disparity 
    Input: lbp
    Output: cost of each disparity 
    '''
    cost_in_diff_disparity_list = []    # Shift right image 
    shift_img = None
    static_img = None
    if shift_direction == 'r':
        shift_img = Ir_lbp
        static_img = Il_lbp
    else: 
        shift_img = Il_lbp
        static_img = Ir_lbp
    ## Right shift right image (0 ~ max_disp)
    # - If left image then shift left 
    for _cur_disp in range(max_disp): 
        shift_result = np.empty_like(shift_img)
        if _cur_disp == 0: 
            shift_result = shift_img
        elif shift_direction == 'r': 
            shift_result[:, _cur_disp:, :] = shift_img[:, :-_cur_disp, :]
        else: 
            shift_result[:, :-_cur_disp, :] = shift_img[:, _cur_disp:, :]
        # print(shift_result[:4, :4, :])

        ## Calculate cost (hamming distance) (a single disparity)
        after_xor = np.logical_xor(static_img, shift_result)
        after_xor_sum = np.sum(after_xor, axis=2)
        if _cur_disp != 0: 
            if shift_direction == 'r': 
                after_xor_sum[:, :_cur_disp] = after_xor_sum[:, _cur_disp:_cur_disp+1] # Cost of closest valid pixel
            else: 
                after_xor_sum[:, -_cur_disp:] = after_xor_sum[:, -_cur_disp-1:-_cur_disp] # Cost of closest valid pixel
        # print(after_xor.shape)
        # print(after_xor_sum.shape)
        # print(after_xor_sum[:10, :10])

        cost_in_diff_disparity_list.append(after_xor_sum)

    ## Reshape the cost to (h, w, max_disparity)
    cost_in_diff_disparity_list = [_arr[:, :, None] for _arr in cost_in_diff_disparity_list]
    cost_disp_arr = np.concatenate(cost_in_diff_disparity_list, axis=2)
    # print(cost_disp_arr.shape)
    
    return cost_disp_arr

def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency

    # ## Shift original image from 0 to max_disp for guided filter (bgr 3 channel image)
    # for _cur_disp in range(max_disp): 


    ## Calculate one color channel's lbp first (3x3 LBP then hamming distance as cost)
    Il_b = Il[:, :, 0]
    Il_g = Il[:, :, 1]
    Il_r = Il[:, :, 2]
    Ir_b = Ir[:, :, 0]
    Ir_g = Ir[:, :, 1]
    Ir_r = Ir[:, :, 2]
    Il_b_lbp = lbpCalculate(Il_b, h, w)
    Il_g_lbp = lbpCalculate(Il_g, h, w)
    Il_r_lbp = lbpCalculate(Il_r, h, w)
    Ir_b_lbp = lbpCalculate(Ir_b, h, w)
    Ir_g_lbp = lbpCalculate(Ir_g, h, w)
    Ir_r_lbp = lbpCalculate(Ir_r, h, w)
    print('left image blue channel shape: ', Il_b.shape)
    print('left image blue channel lbp shape: ', Il_b_lbp.shape)
    print(Il_b[:3, :3])
    print(Ir_b[:5, :5])
    print(Il_b_lbp[1, 1, :])
    print(Ir_b_lbp[1, 1, :])

    ## Compute cost of each disparity of each channel of each pixel 
    shift_right_b_disp = computeCensusCost(Il_b_lbp, Ir_b_lbp, max_disp)
    shift_right_g_disp = computeCensusCost(Il_g_lbp, Ir_g_lbp, max_disp)
    shift_right_r_disp = computeCensusCost(Il_r_lbp, Ir_r_lbp, max_disp)

    shift_left_b_disp = computeCensusCost(Il_b_lbp, Ir_b_lbp, max_disp, 'l')
    shift_left_g_disp = computeCensusCost(Il_g_lbp, Ir_g_lbp, max_disp, 'l')
    shift_left_r_disp = computeCensusCost(Il_r_lbp, Ir_r_lbp, max_disp, 'l')
    print(shift_right_b_disp.shape)

    ## sum up all the cost in each channel (BGR 3 channels)
    shift_right_disp = np.add(np.add(shift_right_b_disp, shift_right_g_disp), shift_right_r_disp).astype(np.float32)
    shift_left_disp = np.add(np.add(shift_left_b_disp, shift_left_g_disp), shift_left_r_disp).astype(np.float32)
    print(shift_right_disp.shape)


    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    guided_shift_right_disp = xip.guidedFilter(Il, shift_right_disp, radius=1, eps=50)
    guided_shift_left_disp = xip.guidedFilter(Ir, shift_left_disp, radius=1, eps=50)
    print(guided_shift_right_disp.shape)


    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    np.argmin(guided_shift_right_disp, axis=2)

    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering (Opencv has implemented)


    return labels.astype(np.uint8)
    