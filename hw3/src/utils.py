import numpy as np


def solve_homography(u, v):
    """
    This function should return a 3-by-3 homography matrix,
    u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
    :param u: N-by-2 source pixel location matrices
    :param v: N-by-2 destination pixel location matrices
    :return:
    """
    N = u.shape[0]
    H = None

    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')

    # I prefer solution 2 
    # TODO: 1.forming A 
    A_row_list = []
    for point_u, point_v in zip(u, v): 
        # print(point_u, ', ', point_v)
        arr1 = np.array([point_u[0], point_u[1], 1, 0, 0, 0, -1 * point_u[0] * point_v[0], -1 * point_u[1] * point_v[0], -1 * point_v[0]])
        arr2 = np.array([0, 0, 0, point_u[0], point_u[1], 1, -1 * point_u[0] * point_v[1], -1 * point_u[1] * point_v[1], -1 * point_v[1]])
        A_row_list.append(arr1)
        A_row_list.append(arr2)
    A_mat = np.vstack(A_row_list)
    # print(A_mat.shape)

    # TODO: 2.solve H with A
    (U, D, V) = np.linalg.svd(A_mat)
    V = np.transpose(V)
    H = V[:, -1]
    H = H.reshape((3, 3))
    # print(U.shape)
    # print(D.shape)
    # print(V.shape)
    # print(H.shape)
    return H


def warping(src, dst, H, ymin, ymax, xmin, xmax, direction='b'):
    """
    Perform forward/backward warpping without for loops. i.e.
    for all pixels in src(xmin~xmax, ymin~ymax),  warp to destination
          (xmin=0,ymin=0)  source                       destination
                         |--------|              |------------------------|
                         |        |              |                        |
                         |        |     warp     |                        |
    forward warp         |        |  --------->  |                        |
                         |        |              |                        |
                         |--------|              |------------------------|
                                 (xmax=w,ymax=h)

    for all pixels in dst(xmin~xmax, ymin~ymax),  sample from source
                            source                       destination
                         |--------|              |------------------------|
                         |        |              | (xmin,ymin)            |
                         |        |     warp     |           |--|         |
    backward warp        |        |  <---------  |           |__|         |
                         |        |              |             (xmax,ymax)|
                         |--------|              |------------------------|

    :param src: source image
    :param dst: destination output image
    :param H:
    :param ymin: lower vertical bound of the destination(source, if forward warp) pixel coordinate
    :param ymax: upper vertical bound of the destination(source, if forward warp) pixel coordinate
    :param xmin: lower horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param xmax: upper horizontal bound of the destination(source, if forward warp) pixel coordinate
    :param direction: indicates backward warping or forward warping
    :return: destination output image
    """

    h_src, w_src, ch = src.shape
    h_dst, w_dst, ch = dst.shape
    H_inv = np.linalg.inv(H)

    # TODO: 1.meshgrid the (x,y) coordinate pairs
    meshgrid_x, meshgrid_y = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))
    # meshgrid_x_3d = meshgrid_x.reshape((meshgrid_x.shape[0], meshgrid_x.shape[1], 1))
    # meshgrid_y_3d = meshgrid_y.reshape((meshgrid_y.shape[0], meshgrid_y.shape[1], 1))
    # meshgrid_xy_3d = np.concatenate((meshgrid_x_3d, meshgrid_y_3d), axis=2)
    # meshgrid_xy_3d_final = meshgrid_xy_3d.reshape((meshgrid_xy_3d.shape[2], meshgrid_xy_3d.shape[0]*meshgrid_xy_3d.shape[1]))
    # print('ymin: ', ymin)
    # print('ymax: ', ymax)
    # print(meshgrid_y)
    # print(meshgrid_y.shape)
    # print(meshgrid_x)
    # print(meshgrid_y)

    # TODO: 2.reshape the destination pixels as N x 3 homogeneous coordinate
    pixels_idx = np.vstack([
        meshgrid_x.reshape(meshgrid_x.shape[0]*meshgrid_x.shape[1]), 
        meshgrid_y.reshape(meshgrid_y.shape[0]*meshgrid_y.shape[1]), 
        np.ones((meshgrid_y.shape[0]*meshgrid_y.shape[1]), dtype=int)
    ])
    print(pixels_idx)


    if direction == 'b':
        # TODO: 3.apply H_inv to the destination pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of source image)

        # TODO: 5.sample the source image with the masked and reshaped transformed coordinates

        # TODO: 6. assign to destination image with proper masking

        pass

    elif direction == 'f':
        # TODO: 3.apply H to the source pixels and retrieve (u,v) pixels, then reshape to (ymax-ymin),(xmax-xmin)
        new_pixels_idx = np.dot(H, pixels_idx)
        print(new_pixels_idx[:, :])
        new_pixels_idx[0, :] = np.divide(new_pixels_idx[0, :], new_pixels_idx[2, :])
        new_pixels_idx[1, :] = np.divide(new_pixels_idx[1, :], new_pixels_idx[2, :])
        new_pixels_idx[2, :] = np.ones_like(new_pixels_idx[2, :])

        print(pixels_idx.shape)
        print(new_pixels_idx.shape)
        print(new_pixels_idx[:, :])

        new_pixels_idx = new_pixels_idx.reshape((3, ymax-ymin, xmax-xmin))
        print(new_pixels_idx[:, 0, 0])

        # TODO: 4.calculate the mask of the transformed coordinate (should not exceed the boundaries of destination image)
        mask = np.ones_like(new_pixels_idx, dtype=bool)
        mask[0, :, :] = (new_pixels_idx[0, :, :] >= 0) & (new_pixels_idx[0, :, :] < w_dst)
        mask[1, :, :] = (new_pixels_idx[1, :, :] >= 0) & (new_pixels_idx[1, :, :] < h_dst)
        new_mask = (mask[0, :, :] & mask[1, :, :])
        # print(new_mask.shape)

        # TODO: 5.filter the valid coordinates using previous obtained mask
        valid_coord_idx = new_pixels_idx[:, new_mask]
        valid_coord_idx = np.round(valid_coord_idx).astype(int)
        valid_coord_idx = valid_coord_idx[:2, :]
        valid_coord_idx = valid_coord_idx.reshape(2, ymax-ymin, xmax-xmin)
        print(valid_coord_idx.shape)
        print(valid_coord_idx)

        # TODO: 6. assign to destination image using advanced array indicing
        # print(dst[valid_coord_idx[0, :, :], valid_coord_idx[1, :, :], :].shape)
        # new_mask = np.transpose(new_mask)
        print(src.shape)
        print(new_mask.shape)
        print(src[new_mask, :].reshape((src.shape[0], src.shape[1], src.shape[2])).shape)
        dst[valid_coord_idx[1, :, :], valid_coord_idx[0, :, :], :] = \
        src[new_mask, :].reshape((src.shape[0], src.shape[1], src.shape[2]))

        pass

    return dst


