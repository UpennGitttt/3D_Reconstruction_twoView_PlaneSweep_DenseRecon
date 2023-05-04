import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import imageio
from tqdm import tqdm
from transforms3d.euler import mat2euler, euler2mat
import pyrender
import trimesh
import cv2
import open3d as o3d


from dataloader import load_middlebury_data

# from utils import viz_camera_poses

EPS = 1e-8


def homo_corners(h, w, H):
    corners_bef = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    corners_aft = cv2.perspectiveTransform(corners_bef, H).squeeze(1)
    u_min, v_min = corners_aft.min(axis=0)
    u_max, v_max = corners_aft.max(axis=0)
    return u_min, u_max, v_min, v_max


def rectify_2view(rgb_i, rgb_j, R_irect, R_jrect, K_i, K_j, u_padding=20, v_padding=20):

    # reference: https://stackoverflow.com/questions/18122444/opencv-warpperspective-how-to-know-destination-image-size
    assert rgb_i.shape == rgb_j.shape, "This hw assumes the input images are in same size"
    h, w = rgb_i.shape[:2]

    ui_min, ui_max, vi_min, vi_max = homo_corners(h, w, K_i @ R_irect @ np.linalg.inv(K_i))
    uj_min, uj_max, vj_min, vj_max = homo_corners(h, w, K_j @ R_jrect @ np.linalg.inv(K_j))

    # The distortion on u direction (the world vertical direction) is minor, ignore this
    w_max = int(np.floor(max(ui_max, uj_max))) - u_padding * 2
    h_max = int(np.floor(min(vi_max - vi_min, vj_max - vj_min))) - v_padding * 2

    assert K_i[0, 2] == K_j[0, 2], "This hw assumes original K has same cx"
    K_i_corr, K_j_corr = K_i.copy(), K_j.copy()
    K_i_corr[0, 2] -= u_padding
    K_i_corr[1, 2] -= vi_min + v_padding
    K_j_corr[0, 2] -= u_padding
    K_j_corr[1, 2] -= vj_min + v_padding

    H_i = K_i_corr @ R_irect @ np.linalg.inv(K_i)  # this is a homography KRK-1
    H_j = K_j_corr @ R_jrect @ np.linalg.inv(K_j)  # this is a homography KRK-1
    rgb_i_rect = cv2.warpPerspective(rgb_i, H_i, dsize=(w_max, h_max))  # wrap all the inner points 
    rgb_j_rect = cv2.warpPerspective(rgb_j, H_j, dsize=(w_max, h_max))



    return rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr


def compute_right2left_transformation(R_wi, T_wi, R_wj, T_wj):
    R_ji = R_wi @ np.linalg.inv(R_wj)
    T_ji = -R_wi @ np.linalg.inv(R_wj) @ T_wj + T_wi
    B = np.linalg.norm(T_ji)

    return R_ji, T_ji, B


def compute_rectification_R(T_ji):
    # check the direction of epipole, should point to the positive direction of y axis
    e_i = T_ji.squeeze(-1) / (T_ji.squeeze(-1)[1] + EPS)
    
    # ! Note, we define a small EPS at the beginning of this file, use it when you normalize each column

    """Student Code Starts"""
    R_irect = np.zeros((3, 3))
    r1 = e_i / np.linalg.norm(e_i) # unit vector of epipole, because epipole = image of the other camera center
    Tx = T_ji[0] # convert to number
    Ty = T_ji[1] # convert to int
    Tz = T_ji[2] # convert to int

    r2_right = np.array([-Ty[0], Tx[0], 0]).reshape(3,) * -1
    # print(Tx, Ty, Tz)
    r2 = (1/np.sqrt((Tx[0]**2) + (Ty[0]**2))) * r2_right

    r3 = np.cross(r2, r1)
    # print(r1, r2, r3.shape)
    # R_irect[0] = np.transpose(r2)
    # R_irect[1] = np.transpose(r1)
    # R_irect[2] = np.transpose(r3)
    R_irect[0] = r2
    R_irect[1] = r1
    R_irect[2] = r3


    return R_irect


def ssd_kernel(src, dst):
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    """Student Code Starts"""
    # print("-------")
    # print(src.shape)
    # intialize the ssd map with all zeros
    sd1 = np.zeros((src.shape[0], dst.shape[0], 3))
    # xx, yy = np.meshgrid(np.arange(3), np.arange(3))
    # go over each channel
    for i in range(src.shape[2]):
        for j in range(src.shape[0]):
            diff = (src[j, :, i] - dst[:, :, i])
            sd1[j, :, i] = np.linalg.norm(diff, axis=1) ** 2
    # summing up the error
    ssd = np.sum(sd1, axis=2)

    """Student Code Ends"""

    return ssd  # M,N


def sad_kernel(src, dst):
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    # print("-------")
    # print(src.shape)
    # intialize the ssd map with all zeros
    sad1 = np.zeros((src.shape[0], dst.shape[0], 3))
    # xx, yy = np.meshgrid(np.arange(3), np.arange(3))
    # go over each channel
    for i in range(src.shape[2]):
        for j in range(src.shape[0]):
            diff = src[j, :, i] - dst[:, :, i]
            sad1[j, :, i] = np.sum(np.abs(diff), axis=1)
    # summing up the error
    sad = np.sum(sad1, axis=2)


    return sad  # M,N


def zncc_kernel(src, dst):
    # src: M,K*K,3; dst: N,K*K,3
    assert src.ndim == 3 and dst.ndim == 3
    assert src.shape[1:] == dst.shape[1:]

    # print("-------")
    # print(src.shape)
    # initialize output
    znc = np.zeros((src.shape[0], dst.shape[0], 3))
    # calculate each element in the formula
    mean_Wsrc = np.mean(src, axis=1)
    sig_src = np.std(src, axis=1)
    # print("----------")
    # print(sig_dst)
    # print(mean_Wdst)
    mean_Wdst = np.mean(dst, axis=1)
    sig_dst = np.std(dst, axis=1)
    # xx, yy = np.meshgrid(np.arange(3), np.arange(3))
    # p1 = sig_src

    # put this all together in the formula
    for i in range(src.shape[2]):
        for j in range(src.shape[0]):
            p1 = sig_src[j, i] * sig_dst[:, i] + EPS
            p2 = (src[j, :, i] - mean_Wsrc[j, i]) * (dst[:, :, i] - mean_Wdst[:, i].reshape(-1, 1))        
            znc[j, :, i] = np.sum(p2, axis=1) / p1
    zncc = np.sum(znc, axis=2)

    # ! note here we use minus zncc since we use argmin outside, but the zncc is a similarity, which should be maximized
    return zncc * (-1.0)  # M,N


def image2patch(image, k_size):

    # first zero padding:
    print("-----")
    padwidth = k_size//2
    print(image.shape) # (475, 611, 3)
    padded_image = np.pad(image, k_size//2, mode='constant')
    padded_image = padded_image[:, :, padwidth:padwidth + 3] # make sure last dimension is still 3
    # k_size = 3
    

    # print("++++++")
    # print(padded_image.shape) # (477, 613, 5)

    # initialize output
    patch_buffer = np.zeros((image.shape[0], image.shape[1], k_size ** 2, 3))
    H = image.shape[0]
    W = image.shape[1]


    xx, yy = np.meshgrid(np.arange(padwidth, H + padwidth), np.arange(padwidth, W+padwidth))


    def findcoord(x, y, ):
        x_pad = x + padwidth
        y_pad = y + padwidth
        x_img, y_img = np.meshgrid(np.arange(x, (x_pad+padwidth+1)), np.arange(y , (y_pad+padwidth+1)))

        return x_img, y_img

    # patch_buffer[yy_buffer.ravel(), xx_buffer.ravel()] = padded_image[yy.ravel(), xx.ravel()]
    
    for x in range(W):  #(475, 611, 3) H = 475, W = 611
        for y in range(H):
            x_img, y_img = findcoord(x, y)
            for z in range(3):
                individual_patch = padded_image[y_img, x_img, z].ravel()
                patch_buffer[y, x, :, z] = individual_patch


    return patch_buffer  # H,W,K**2,3


def compute_disparity_map(
    rgb_i, rgb_j, d0, k_size=5, kernel_func=ssd_kernel, img2patch_func=image2patch):

    # print(rgb_i.shape)
    # print(rgb_j.shape)
    row, col = rgb_j.shape[:2]
    # initialize output with type float
    disp_map = np.zeros((row, col), dtype=np.float64)
    lr_consistency_mask = np.zeros((row, col), dtype=np.float64)
    # find patches
    left_patch = image2patch(rgb_i.astype(float) / 255.0, k_size) 
    right_patch = image2patch(rgb_j.astype(float) / 255.0, k_size)  
    # print(left_patch.shape, right_patch.shape)
   
    # vectorize the step
    xx, yy = np.arange(row), np.arange(row)
    # print(xx)
    all_disp = xx[:, None] - yy[None, :] + d0
    # chosen_disp = np.where(all_disp > 0.0)
    chosen_disp = all_disp > 0.0


    for i in range(col):
        # get column
        left_small, right_small = left_patch[:, i], right_patch[:, i]
        # print(left_small.shape, right_small.shape)

        # using kernel function to calculate the diff
        kval = kernel_func(left_small, right_small)
        max_k = kval.max() + 1.0
        kval[~chosen_disp] = max_k

        right_match = np.argmin(kval, axis=1)
        left_match = np.argmin(kval[:, right_match], axis=0)

        flag = left_match == np.arange(row)
        
        # print(flag)

        lr_consistency_mask[:, i] = flag
        # print(lr_consistency_mask)

        vL = np.arange(row)
        vR = right_match
        # d0 + vL - vR
        d = d0 + vL - vR
        disp_map[:, i] = d

    disp_map = disp_map.astype(np.float64)
    lr_consistency_mask = lr_consistency_mask.astype(np.float64)


    return disp_map, lr_consistency_mask


def compute_dep_and_pcl(disp_map, B, K):
    # initialization
    row, col = disp_map.shape    
    # initialize the each individual point coordinate
    individual = np.zeros((3, row * col))
    # # using meshgrid for vectorizing
    xx, yy = np.meshgrid(np.arange(col), np.arange(row))
    dep_map = B * K[1,1] / disp_map
    # individual[0, :] = xx.ravel()
    # individual[1, :] = yy.ravel()
    individual[2, :] = 1
    # get each point's coord [x, y, 1].T
    coor = np.vstack((xx.ravel(), yy.ravel(), individual[2, :])) 
    calib_coor  = np.linalg.inv(K) @ coor
    # using formula to calculate xyz_cam
    xyz_cam = calib_coor * dep_map.ravel()
    xyz_cam = xyz_cam.T
    # reshape to the shape as directed
    xyz_cam = xyz_cam.reshape(row, col, 3)


    return dep_map, xyz_cam


def postprocess(
    dep_map,
    rgb,
    xyz_cam,
    R_wc,
    T_wc,
    consistency_mask=None,
    hsv_th=45,
    hsv_close_ksize=11,
    z_near=0.45,
    z_far=0.65,
):
    # extract mask from rgb to remove background
    mask_hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., -1]
    mask_hsv = (mask_hsv > hsv_th).astype(np.uint8) * 255
    # imageio.imsave("./debug_hsv_mask.png", mask_hsv)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (hsv_close_ksize, hsv_close_ksize))
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, morph_kernel).astype(float)
    # imageio.imsave("./debug_hsv_mask_closed.png", mask_hsv)

    # constraint z-near, z-far
    mask_dep = ((dep_map > z_near) * (dep_map < z_far)).astype(float)
    # imageio.imsave("./debug_dep_mask.png", mask_dep)

    mask = np.minimum(mask_dep, mask_hsv)
    if consistency_mask is not None:
        mask = np.minimum(mask, consistency_mask)
    # imageio.imsave("./debug_before_xyz_mask.png", mask)

    # filter xyz point cloud
    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcl_cam.reshape(-1, 3).copy())
    cl, ind = o3d_pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=2.0)
    _pcl_mask = np.zeros(pcl_cam.shape[0])
    _pcl_mask[ind] = 1.0
    pcl_mask = np.zeros(xyz_cam.shape[0] * xyz_cam.shape[1])
    pcl_mask[mask.reshape(-1) > 0] = _pcl_mask
    mask_pcl = pcl_mask.reshape(xyz_cam.shape[0], xyz_cam.shape[1])
    # imageio.imsave("./debug_pcl_mask.png", mask_pcl)
    mask = np.minimum(mask, mask_pcl)
    # imageio.imsave("./debug_final_mask.png", mask)

    pcl_cam = xyz_cam.reshape(-1, 3)[mask.reshape(-1) > 0]
    pcl_color = rgb.reshape(-1, 3)[mask.reshape(-1) > 0]

    
    pcl_world = (np.linalg.inv(R_wc) @ pcl_cam.T - np.linalg.inv(R_wc) @ T_wc).T


    return mask, pcl_world, pcl_cam, pcl_color


def two_view(view_i, view_j, k_size=5, kernel_func=ssd_kernel):
    # Full pipeline

    # * 1. rectify the views
    R_wi, T_wi = view_i["R"], view_i["T"][:, None]  # p_i = R_wi @ p_w + T_wi
    R_wj, T_wj = view_j["R"], view_j["T"][:, None]  # p_j = R_wj @ p_w + T_wj

    R_ji, T_ji, B = compute_right2left_transformation(R_wi, T_wi, R_wj, T_wj)
    assert T_ji[1, 0] > 0, "here we assume view i should be on the left, not on the right"

    R_irect = compute_rectification_R(T_ji)

    rgb_i_rect, rgb_j_rect, K_i_corr, K_j_corr = rectify_2view(
        view_i["rgb"],
        view_j["rgb"],
        R_irect,
        R_irect @ R_ji,
        view_i["K"],
        view_j["K"],
        u_padding=20,
        v_padding=20,
    )

    # * 2. compute disparity
    assert K_i_corr[1, 1] == K_j_corr[1, 1], "This hw assumes the same focal Y length"
    assert (K_i_corr[0] == K_j_corr[0]).all(), "This hw assumes the same K on X dim"
    assert (
        rgb_i_rect.shape == rgb_j_rect.shape
    ), "This hw makes rectified two views to have the same shape"
    disp_map, consistency_mask = compute_disparity_map(
        rgb_i_rect,
        rgb_j_rect,
        d0=K_j_corr[1, 2] - K_i_corr[1, 2],
        k_size=k_size,
        kernel_func=kernel_func,
    )

    # * 3. compute depth map and filter them
    dep_map, xyz_cam = compute_dep_and_pcl(disp_map, B, K_i_corr)

    mask, pcl_world, pcl_cam, pcl_color = postprocess(
        dep_map,
        rgb_i_rect,
        xyz_cam,
        R_wc=R_irect @ R_wi,
        T_wc=R_irect @ T_wi,
        consistency_mask=consistency_mask,
        z_near=0.5,
        z_far=0.6,
    )

    return pcl_world, pcl_color, disp_map, dep_map


def main():
    DATA = load_middlebury_data("data/templeRing")
    # viz_camera_poses(DATA)
    two_view(DATA[0], DATA[3], 5, zncc_kernel)

    return


if __name__ == "__main__":
    main()
