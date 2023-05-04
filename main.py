import numpy as np
import matplotlib.pyplot as plt
import cv2
from utils import viz_camera_poses
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
import k3d
import open3d as o3d
from utils import viz_camera_poses
from dataloader import load_middlebury_data
from two_view_stereo import image2patch, ssd_kernel
from two_view_stereo import compute_disparity_map
from two_view_stereo import compute_dep_and_pcl
from two_view_stereo import postprocess
from two_view_stereo import two_view, ssd_kernel, sad_kernel, zncc_kernel


DATA = load_middlebury_data("data/templeRing")
view_i, view_j = DATA[0], DATA[3]

def viz_3d_embedded(pcl, color):
    plot = k3d.plot(camera_auto_fit=True)
    color = color.astype(np.uint8)
    color32 = (color[:, 0] * 256**2 + color[:, 1] * 256**1 + color[:, 2] * 256**0).astype(
        np.uint32
    )
    plot += k3d.points(pcl.astype(float), color32, point_size=0.001, shader="flat")
    plot.display()


def main():
    from two_view_stereo import (
        rectify_2view,
        compute_rectification_R,
        compute_right2left_transformation,
    )

    R_wi, T_wi = view_i["R"], view_i["T"][:, None]  # p_i = R_wi @ p_w + T_wi
    R_wj, T_wj = view_j["R"], view_j["T"][:, None]  # p_j = R_wj @ p_w + T_wj

    R_ji, T_ji, B = compute_right2left_transformation(R_wi, T_wi, R_wj, T_wj)
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

    # plt.subplot(2, 2, 1)
    # plt.title("input view i")
    # plt.imshow(cv2.rotate(view_i["rgb"], cv2.ROTATE_90_COUNTERCLOCKWISE))
    # plt.subplot(2, 2, 2)
    # plt.title("input view j")
    # plt.imshow(cv2.rotate(view_j["rgb"], cv2.ROTATE_90_COUNTERCLOCKWISE))
    # plt.subplot(2, 2, 3)
    # plt.title("rect view i")
    # plt.imshow(cv2.rotate(rgb_i_rect, cv2.ROTATE_90_COUNTERCLOCKWISE))
    # plt.subplot(2, 2, 4)
    # plt.title("rect view j")
    # plt.imshow(cv2.rotate(rgb_j_rect, cv2.ROTATE_90_COUNTERCLOCKWISE))
    # plt.tight_layout()
    # plt.show()



    h, w = rgb_i_rect.shape[:2]
    d0 = K_j_corr[1, 2] - K_i_corr[1, 2]

    patches_i = image2patch(rgb_i_rect.astype(float) / 255.0, 3)  # [h,w,k*k,3]
    patches_j = image2patch(rgb_j_rect.astype(float) / 255.0, 3)  # [h,w,k*k,3]

    vi_idx, vj_idx = np.arange(h), np.arange(h)
    disp_candidates = vi_idx[:, None] - vj_idx[None, :] + d0
    valid_disp_mask = disp_candidates > 0.0
    u = 400

    buf_i, buf_j = patches_i[:, u], patches_j[:, u]
    value = ssd_kernel(buf_i, buf_j) 
    _upper = value.max() + 1.0
    value[~valid_disp_mask] = _upper
    disp_map, consistency_mask = compute_disparity_map(rgb_i_rect, rgb_j_rect, d0=K_j_corr[1, 2] - K_i_corr[1, 2], k_size=5)

    plt.imshow(cv2.rotate(consistency_mask, cv2.ROTATE_90_COUNTERCLOCKWISE))
    plt.title("L-R consistency check mask")
    plt.show()

    dep_map, xyz_cam = compute_dep_and_pcl(disp_map, B, K_i_corr)
    mask, pcl_world, pcl_cam, pcl_color = postprocess(
    dep_map,
    rgb_i_rect,
    xyz_cam,
    R_wc=R_irect @ R_wi,
    T_wc=R_irect @ T_wi,
    consistency_mask=consistency_mask,
    z_near=0.5,
    z_far=0.6,)

    mask = (mask > 0).astype(np.float)

    viz_3d_embedded(pcl_world, pcl_color.astype(np.uint8))
    pcl_sad, pcl_color_sad, disp_map_sad, dep_map_sad = two_view(DATA[0], DATA[2], 5, sad_kernel)
    viz_3d_embedded(pcl_sad, pcl_color_sad.astype(np.uint8))
    pcl_zncc, pcl_color_zncc, disp_map_zncc, dep_map_zncc = two_view(DATA[0], DATA[2], 5, zncc_kernel)
    viz_3d_embedded(pcl_zncc, pcl_color_zncc.astype(np.uint8))

    pcl_list, pcl_color_list, disp_map_list, dep_map_list = [], [], [], []
    pairs = [(0, 2), (2, 4), (5, 7), (8, 10), (13, 15), (16, 18), (19, 21), (22, 24), (25, 27)]
    # for i in range(13, 28, 3):
    for pair in pairs:
        i,j = pair
        _pcl, _pcl_color, _disp_map, _dep_map = two_view(DATA[i], DATA[j], 5, sad_kernel)
        pcl_list.append(_pcl)
        pcl_color_list.append(_pcl_color)
        disp_map_list.append(_disp_map)
        dep_map_list.append(_dep_map)
    plot = k3d.plot(camera_auto_fit=True)
    for pcl, color in zip(pcl_list, pcl_color_list):
        color = color.astype(np.uint8)
        color32 = (color[:, 0] * 256**2 + color[:, 1] * 256**1 + color[:, 2] * 256**0).astype(
            np.uint32
        )
        plot += k3d.points(pcl.astype(float), color32, point_size=0.001, shader="flat")
    plot.display()


if __name__ == '__main__':
    main()
