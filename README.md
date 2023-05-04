# 3D_Reconstruction_twoView_PlaneSweep_DenseRecon
## This code provides a method and esseential functions such as depth estimation, calculate dispatity, two-view rectification etc. to perform stereo rectification and depth estimation on a pair of images captured by two cameras. This is a course project in Upenn Course CIS 580, dataset is created  by Steve Seitz, James Diebel, Daniel Scharstein, Brian Curless, and Rick Szeliski

## Dependencies
1. numpy
2. matplotlib
3. os
4. imageio
5. tqdm
6. transforms3d
7. pyrender
8. trimesh
9. cv2
10. open3d

# To install the dependencies, run the following command:

pip install numpy matplotlib os imageio tqdm transforms3d pyrender trimesh cv2 open3d

1. homo_corners: Calculates the minimum and maximum values of the corners after applying a homography to an image.

2. rectify_2view: Rectifies two images based on their respective camera poses, intrinsic parameters, and homography matrices.

3. compute_right2left_transformation: Computes the transformation between two camera views, returning the rotation, translation, and baseline.

4. compute_rectification_R: Computes the rectification matrix for the epipolar geometry.

5. ssd_kernel: Computes the sum of squared differences (SSD) between two image patches.

5. sad_kernel: Computes the sum of absolute differences (SAD) between two image patches.

7. zncc_kernel: Computes the zero-mean normalized cross-correlation (ZNCC) between two image patches.

8. image2patch: Converts an image into a set of patches.

9. compute_disparity_map: Computes the disparity map between two rectified images using a specified kernel function.

10. compute_dep_and_pcl: Computes the depth map and point cloud from the disparity map, baseline, and intrinsic camera parameters.

11. postprocess: Post-processes the depth map and point cloud by applying various filters and masking techniques.


## Final Results:

# SSD Kernel:

<img width="456" alt="Screen Shot 2023-05-04 at 6 37 00 PM" src="https://user-images.githubusercontent.com/98191838/236344225-5fabcf58-68c4-415b-be0f-5b9955bd0f9f.png">

# SAD Kernel:

<img width="416" alt="Screen Shot 2023-05-04 at 6 37 30 PM" src="https://user-images.githubusercontent.com/98191838/236344294-a7a5d183-6312-47f8-aee7-d88341fc40d8.png">

# ZNCC Kernel:

<img width="471" alt="Screen Shot 2023-05-04 at 6 37 52 PM" src="https://user-images.githubusercontent.com/98191838/236344346-330b8b6f-fdc4-493d-b77a-574003e84dce.png">

# Aggregated 3D Dense Reconstruction:

<img width="452" alt="Screen Shot 2023-05-04 at 6 38 16 PM" src="https://user-images.githubusercontent.com/98191838/236344399-475c862d-a3a4-4ad4-9a62-c9ebf2b14b88.png">
