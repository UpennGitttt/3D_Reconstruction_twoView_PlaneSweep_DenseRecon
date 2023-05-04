import numpy as np
import cv2


EPS = 1e-8


def backproject_corners(K, width, height, depth, Rt):
    """
    Backproject 4 corner points in image plane to the imaginary depth plane using intrinsics and extrinsics
    
    Hint:
    depth * corners = K @ T @ y, where y is the output world coordinates and T is the 4x4 matrix of Rt (3x4)

    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        width -- width of the image
        heigh -- height of the image
        depth -- depth value of the imaginary plane
    Output:
        points -- 2 x 2 x 3 array of 3D coordinates of backprojected points, here 2x2 correspinds to 4 corners
    """

    points = np.array(
        (
            (0, 0, 1),
            (width, 0, 1),
            (0, height, 1),
            (width, height, 1),
        ),
        dtype=np.float32,
    )

    """ YOUR CODE HERE
    """
    #  depth * corners = K @ T @ y, where y is the output world coordinates and T is the 4x4 matrix of Rt (3x4)
    # Rt -- 3 x 4 camera extrinsics calibration matrix
    last = np.array([0, 0, 0, 1])
    ones = np.array([1, 1, 1, 1])
    # calculate T
    T = np.vstack((Rt,last))
    out = depth * np.linalg.inv(K) @ points.T
    coor = np.vstack((out, ones))
    points = (np.linalg.inv(T) @ coor).T[:,:3].reshape(2,2,3)
    """ END YOUR CODE
    """
    return points


def project_points(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    
    Hint:
    Z * projections = K @ T @ p, where p is the input points and projections is the output, T is the 4x4 matrix of Rt (3x4)
    
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- points_height x points_width x 3 array of 3D points
    Output:
        projections -- points_height x points_width x 2 array of 2D projections
    """
    """ YOUR CODE HERE
    """
    # Rt -- 3 x 4 camera extrinsics calibration matrix
    last = np.array([0, 0, 0, 1])
    # calculate T
    T = np.vstack((Rt,last))
    p = np.vstack((points.reshape(4,3).T, np.ones((1,4))))
    # Z * projections = K @ T @ p
    points = K @ (T @ p)[0:3,:]
    points = points / points[-1,:].reshape(4,)
    points = points.T.reshape(2,2,3)

    """ END YOUR CODE
    """
    return points[:, :, 0:2]


def warp_neighbor_to_ref(
    backproject_fn, project_fn, depth, neighbor_rgb, K_ref, Rt_ref, K_neighbor, Rt_neighbor
):
    """
    Warp the neighbor view into the reference view
    via the homography induced by the imaginary depth plane at the specified depth

    Make use of the functions you've implemented in the same file (which are passed in as arguments):
    - backproject_corners
    - project_points

    Also make use of the cv2 functions:
    - cv2.findHomography
    - cv2.warpPerspective
    
    ! Note, when you use cv2.warpPerspective, you should use the shape (width, height), NOT (height, width)
    
    Hint: you should do the follows:
    1.) apply backproject_corners on ref view to get the virtual 3D corner points in the virtual plane
    2.) apply project_fn to project these virtual 3D corner points back to ref and neighbor views
    3.) use findHomography to get teh H between neighbor and ref
    4.) warp the neighbor view into the reference view

    Input:
        backproject_fn -- backproject_corners function
        project_fn -- project_points function
        depth -- scalar value of the depth at the imaginary depth plane
        neighbor_rgb -- height x width x 3 array of neighbor rgb image
        K_ref -- 3 x 3 camera intrinsics calibration matrix of reference view
        Rt_ref -- 3 x 4 camera extrinsics calibration matrix of reference view
        K_neighbor -- 3 x 3 camera intrinsics calibration matrix of neighbor view
        Rt_neighbor -- 3 x 4 camera extrinsics calibration matrix of neighbor view
    Output:
        warped_neighbor -- height x width x 3 array of the warped neighbor RGB image
    """

    height, width = neighbor_rgb.shape[:2]
    

    """ YOUR CODE HERE
    """
    points = np.array(
        (
            (0, 0, 1),
            (width, 0, 1),
            (0, height, 1),
            (width, height, 1),
        ),
        dtype=np.float32,
    )
    # find correspondence
    corners_points = backproject_fn(K_ref, width, height, depth, Rt_ref)
    projection_points = project_fn(K_neighbor, Rt_neighbor, corners_points).reshape(4,2)

    H, _ = cv2.findHomography(projection_points, points)
    warped_neighbor = cv2.warpPerspective(neighbor_rgb, H, dsize=(width, height))


    """ END YOUR CODE
    """
    return warped_neighbor


def zncc_kernel_2D(src, dst):
    """
    Compute the cost map between src and dst patchified images via the ZNCC metric

    IMPORTANT NOTES:
    - Treat each RGB channel separately but sum the 3 different zncc scores at each pixel

    - When normalizing by the standard deviation, add the provided small epsilon value,
    EPS which is included in this file, to both sigma_src and sigma_dst to avoid divide-by-zero issues

    Input:
        src -- height x width x K**2 x 3
        dst -- height x width x K**2 x 3
    Output:
        zncc -- height x width array of zncc metric computed at each pixel
    """
    assert src.ndim == 4 and dst.ndim == 4
    assert src.shape[:] == dst.shape[:]

    """ YOUR CODE HERE
    """
    row, col, dim = src.shape[:3]
    zncc = np.zeros((row, col))

    dst = dst.transpose(2,0,1,3).reshape(dim, row * col, 3)
    src = src.transpose(2,0,1,3).reshape(dim, row * col, 3)

    w_src1 = np.mean(src, axis = 0)
    w_src2 = np.std(src, axis = 0)

    w_dst1 = np.mean(dst, axis = 0)
    w_dst2 = np.std(dst, axis = 0)

    for i in range(3):
        numerator = np.sum((src[:, :, i] - w_src1[:, i])*(dst[:, :, i] - w_dst1[:, i]),axis = 0)
        denominator = w_src2[:, i] * w_dst2[:, i] + EPS
        zncc += (numerator / denominator).reshape(row, col)
        # zncc[:, :] += (numerator / denominator).reshape(row, col)

    """ END YOUR CODE
    """

    return zncc  # height x width


def backproject(dep_map, K):
    """
    Backproject image points to 3D coordinates wrt the camera frame according to the depth map

    Input:
        K -- camera intrinsics calibration matrix
        dep_map -- height x width array of depth values
    Output:
        points -- height x width x 3 array of 3D coordinates of backprojected points
    """
    _u, _v = np.meshgrid(np.arange(dep_map.shape[1]), np.arange(dep_map.shape[0]))

    """ YOUR CODE HERE
    """
    # construct all coordinate
    coor = np.vstack((_u.ravel(), _v.ravel(), np.ones(_u.ravel().shape)))
    xyz_cam = ((np.linalg.inv(K) @ coor) * dep_map.ravel()).T
    xyz_cam = xyz_cam.reshape(dep_map.shape[0], dep_map.shape[1], 3)

    """ END YOUR CODE
    """
    return xyz_cam
