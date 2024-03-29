import numpy as np
import cv2.ximgproc as xip
import cv2
# import 

def lbpCalculate(imgIn, h, w): 
    '''
    :imgIn: gray scale image
    :return: lbp array (h, w, 8)
    '''
    imgInPad = cv2.copyMakeBorder(imgIn, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 255)
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

    ## Shift original image from 0 to max_disp for guided filter (bgr 3 channel image)


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
    # print('left image blue channel shape: ', Il_b.shape)
    # print('left image blue channel lbp shape: ', Il_b_lbp.shape)
    # print(Il_b[:3, :3])
    # print(Ir_b[:5, :5])
    # print(Il_b_lbp[1, 1, :])
    # print(Ir_b_lbp[1, 1, :])

    ## Compute cost of each disparity of each channel of each pixel 
    shift_right_b_disp = computeCensusCost(Il_b_lbp, Ir_b_lbp, max_disp)
    shift_right_g_disp = computeCensusCost(Il_g_lbp, Ir_g_lbp, max_disp)
    shift_right_r_disp = computeCensusCost(Il_r_lbp, Ir_r_lbp, max_disp)

    shift_left_b_disp = computeCensusCost(Il_b_lbp, Ir_b_lbp, max_disp, 'l')
    shift_left_g_disp = computeCensusCost(Il_g_lbp, Ir_g_lbp, max_disp, 'l')
    shift_left_r_disp = computeCensusCost(Il_r_lbp, Ir_r_lbp, max_disp, 'l')
    # print(shift_right_b_disp.shape)

    ## sum up all the cost in each channel (BGR 3 channels)
    shift_right_disp = np.add(np.add(shift_right_b_disp, shift_right_g_disp), shift_right_r_disp).astype(np.float32)
    shift_left_disp = np.add(np.add(shift_left_b_disp, shift_left_g_disp), shift_left_r_disp).astype(np.float32)
    # print(shift_right_disp.shape)


    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    # Ref: https://jinzhangyu.github.io/2018/09/06/2018-09-06-OpenCV-Python%E6%95%99%E7%A8%8B-16-%E5%B9%B3%E6%BB%91%E5%9B%BE%E5%83%8F-3/ 
    guided_shift_right_disp = xip.guidedFilter(Il, shift_right_disp, radius=10, eps=100)
    guided_shift_left_disp = xip.guidedFilter(Ir, shift_left_disp, radius=10, eps=100)
    # print('After guided shape: ', guided_shift_right_disp.shape)


    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    shift_right_WTA = np.argmin(guided_shift_right_disp, axis=2)
    shift_left_WTA = np.argmin(guided_shift_left_disp, axis=2)
    disparity_left = shift_right_WTA
    disparity_right = shift_left_WTA
    # print('After WTA shape: ', shift_right_WTA.shape)

    # Output a testing image, checking if codes above works fine
    # norm_sh_right_wta = cv2.normalize(shift_right_WTA, None, 0, 255, cv2.NORM_MINMAX)
    # cv2.imwrite('./test_right.png', norm_sh_right_wta)
    # norm_sh_left_wta = np.zeros_like(shift_left_WTA)
    # norm_sh_left_wta = cv2.normalize(shift_left_WTA, None, 0, 255, cv2.NORM_MINMAX)
    # cv2.imwrite('./test_left.png', norm_sh_left_wta)
    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering (Opencv has implemented)

    ## Check left and right consistency, if different then make hole
    x_idx, y_idx = np.meshgrid(np.arange(w), np.arange(h))
    xy_idx = np.vstack([x_idx.flatten(), y_idx.flatten()])
    x_minus_disparity_idx = xy_idx[0, :] - disparity_left[xy_idx[1, :], xy_idx[0, :]]
    disparity_right_minus_disp = disparity_right[xy_idx[1, :], x_minus_disparity_idx].reshape(h, w)
    # - Make different disparity pixels value to max_disp + 5
    diff_map = (disparity_left == disparity_right_minus_disp)
    # print(diff_map.shape)
    disparity_left_with_holes = np.copy(disparity_left)
    disparity_left_with_holes[~diff_map] = max_disp + 5

    ### Test right image after minus disparity 
    # norm_right_dis = cv2.normalize(disparity_right_minus_disp, None, 0, 255, cv2.NORM_MINMAX)
    # cv2.imwrite('./test_right_after_disparity.png', norm_right_dis)
    ### Test right image after minus disparity 
    # norm_left_hole_dis = cv2.normalize(disparity_left_with_holes, None, 0, 255, cv2.NORM_MINMAX)
    # cv2.imwrite('./test_left_after_holes.png', norm_left_hole_dis)

    ## Hole filling
    ### Bounding point pad by maximum value = max_disp 
    # - If pixel at board and it's a hole, then fill it with max_disp
    x_board = (x_idx==0) | (x_idx==w-1)
    y_board = (y_idx==0) | (y_idx==h-1)
    board_and_hole_bool = (x_board | y_board) & (~diff_map)
    disparity_left_with_holes[board_and_hole_bool] = max_disp
    
    ### Fill the hole from left (first valid pixels disparity)
    not_board_and_hole_bool = ~(x_board | y_board) & (~diff_map)
    # print(not_board_and_hole_bool)
    # print(not_board_and_hole_bool.sum())

    #### Shift to right == find valid from left
    disparity_left_fill_from_left = np.copy(disparity_left_with_holes)
    cur_shifted_not_board_and_hole_bool = np.copy(not_board_and_hole_bool)
    for _shift in range(1, w): 
        # - If all pixel after shift is valid (False), then break 
        if cur_shifted_not_board_and_hole_bool.sum() == 0: 
            # print('last shift: ', _shift)
            break
        # - Shift valid bool map 
        pre_shifted_not_board_and_hole_bool = np.copy(cur_shifted_not_board_and_hole_bool)
        cur_shifted_not_board_and_hole_bool[:, 1:] = np.logical_and(cur_shifted_not_board_and_hole_bool[:, 1:], cur_shifted_not_board_and_hole_bool[:, :-1])
        cur_shifted_not_board_and_hole_bool[:, 0] = False
        # - Check if any pixel change boolean value, if so, 
        #   change their corresponding disparity map value by shifted disparity map 
        # shifted_holes_bool = np.logical_xor(cur_shifted_not_board_and_hole_bool[:, 1:], cur_shifted_not_board_and_hole_bool[:, :-1])
        shifted_holes_bool = np.logical_xor(pre_shifted_not_board_and_hole_bool, cur_shifted_not_board_and_hole_bool)
        cur_shift_filled_holes_idx = np.where(shifted_holes_bool==True)
        disparity_left_fill_from_left[cur_shift_filled_holes_idx[0], cur_shift_filled_holes_idx[1]] = \
            disparity_left_fill_from_left[cur_shift_filled_holes_idx[0], cur_shift_filled_holes_idx[1]-_shift]
        # print(shifted_holes_bool.shape)
        # print(disparity_left_with_holes.shape)
        # print(cur_shift_filled_holes_idx)
        # print(shifted_holes_bool[cur_shift_filled_holes_idx[0], cur_shift_filled_holes_idx[1]])
        # print(disparity_left_with_holes[cur_shift_filled_holes_idx[0], cur_shift_filled_holes_idx[1]])
        # print('Filled holes after shift: ', shifted_holes_bool.sum())

    ### Test after hole filling from left
    # dl_fl = cv2.normalize(disparity_left_fill_from_left, None, 0, 255, cv2.NORM_MINMAX)
    # cv2.imwrite('./test_disp_left_fill_hole_from_left.png', dl_fl)

    ### Fill the hole from right 
    #### Shift to left == find valid from right
    disparity_left_fill_from_right = np.copy(disparity_left_with_holes)
    cur_shifted_not_board_and_hole_bool = np.copy(not_board_and_hole_bool)
    for _shift in range(1, w): 
        # - If all pixel after shift is valid (False), then break 
        if cur_shifted_not_board_and_hole_bool.sum() == 0: 
            # print('last shift: ', _shift)
            break
        # - Shift valid bool map 
        pre_shifted_not_board_and_hole_bool = np.copy(cur_shifted_not_board_and_hole_bool)
        cur_shifted_not_board_and_hole_bool[:, :-1] = np.logical_and(cur_shifted_not_board_and_hole_bool[:, 1:], cur_shifted_not_board_and_hole_bool[:, :-1])
        cur_shifted_not_board_and_hole_bool[:, -1] = False
        # - Check if any pixel change boolean value, if so, 
        #   change their corresponding disparity map value by shifted disparity map 
        shifted_holes_bool = np.logical_xor(pre_shifted_not_board_and_hole_bool, cur_shifted_not_board_and_hole_bool)
        cur_shift_filled_holes_idx = np.where(shifted_holes_bool==True)
        disparity_left_fill_from_right[cur_shift_filled_holes_idx[0], cur_shift_filled_holes_idx[1]] = \
            disparity_left_fill_from_right[cur_shift_filled_holes_idx[0], cur_shift_filled_holes_idx[1]-_shift]

    ### Find minimum from "hole filling from left" and "hole filling from right"
    disparity_left_fill_from_left = disparity_left_fill_from_left[:, :, None]
    disparity_left_fill_from_right = disparity_left_fill_from_right[:, :, None]
    disparity_left_fill_both = np.concatenate((disparity_left_fill_from_left, disparity_left_fill_from_right), axis=2)
    disparity_left_fill_both_min = np.min(disparity_left_fill_both, axis=2)
    disparity_left_with_holes[not_board_and_hole_bool] = disparity_left_fill_both_min[not_board_and_hole_bool]
    disparity_left_holes_filled = disparity_left_with_holes
    # print(disparity_left_fill_from_left.shape)
    # print(disparity_left_fill_from_right.shape)
    # print(disparity_left_fill_both.shape)
    # print(disparity_left_fill_both_min.shape)
    # print((disparity_left_holes_filled==20).sum())
    
    ### Test after hole filling 
    # dl_fl_fin = cv2.normalize(disparity_left_with_holes, None, 0, 255, cv2.NORM_MINMAX)
    # cv2.imwrite('./test_disp_left_hole_filled.png', dl_fl_fin)

    ## Weighted median filtering
    Il_gray = cv2.cvtColor(Il, 6)   #   cv::COLOR_BGR2GRAY = 6, cv::COLOR_RGB2GRAY = 7
    Il_gray = Il_gray.astype(np.uint8)
    disparity_left_holes_filled = disparity_left_holes_filled.astype(np.uint8)
    # print(disparity_left_holes_filled.dtype)
    # print(Il_gray.dtype)
    disparity_left_WMF = xip.weightedMedianFilter(Il_gray, disparity_left_holes_filled, r=5)
    ### Test after WMF
    # dl_wmf = cv2.normalize(disparity_left_WMF, None, 0, 255, cv2.NORM_MINMAX)
    # cv2.imwrite('./test_disp_left_wmf.png', dl_wmf)

    labels = disparity_left_WMF
    return labels.astype(np.uint8)
    