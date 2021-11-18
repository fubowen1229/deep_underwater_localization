import cv2
import numpy as np
import os
import json

import _init_paths
from meshply import MeshPly
from misc_utils import get_3D_corners, solve_pnp, get_camera_intrinsic
import args

if __name__ == '__main__':
    # Path
    test_file = '/data4/fbw/deep_underwater_localization/data/my_data/pool_test.txt'
    bop_root = '/data5/tangjw/underwater/underwater_dataset/underwater_bop'
    bop_test_pth = os.path.join(bop_root, 'test')
    mesh_pth = os.path.join(bop_root, 'models', 'obj_000001.ply')
    
    # Acquire 3D corners
    mesh = MeshPly(mesh_pth)
    vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
    corner_3d = get_3D_corners(vertices)
    corner_3d = corner_3d[0: 3, :].T.reshape(-1, 1, 3) # 8 * 1 * 3

    # Acquire camera matrix
    intrinsics = get_camera_intrinsic()

    # Acquire 2D points
    points_2d = {}
    with open(test_file, 'r') as f:        
        lines = f.readlines()
        for line in lines:
            info = line.strip().split(" ")
            image_id = info[0]
            points_2d[image_id] = {}

            points_2d[image_id]['image_pth'] = info[1]
            points_2d[image_id]['image_size'] = [int(i) for i in info[2:4]]
            points_2d[image_id]['label'] = int(info[4]) # always 0 (only 1 class)
            points_2d[image_id]['corner_2d'] = [int(i) for i in info[5:21]] # different from train

    # Calculate pose
    scene_gt = {}
    for key in points_2d.keys():
        if int(key) != 2600:
            continue
        corner_2d = np.array(points_2d[key]['corner_2d'], dtype=np.double).reshape(-1, 1, 2) # 8 * 1 * 2
        #import ipdb;ipdb.set_trace()
        retval, rot, trans, inliers = cv2.solvePnPRansac(corner_3d, corner_2d, intrinsics, None, flags=cv2.SOLVEPNP_EPNP)
        R = cv2.Rodrigues(rot)[0] # 3 * 3
        T = trans.reshape(-1, 1) * 1000.0 # 3 * 1
        pose = np.concatenate((R, T), 1) # 3 * 4
        points_2d[key]['pose'] = pose

        # Prepare for writing scene_gt.json
        key = int(key)
        scene_gt[key] = []
        gt_pose = {}
        gt_pose["cam_R_m2c"] = R.reshape(9,).tolist()
        gt_pose["cam_t_m2c"] = T.reshape(3,).tolist()
        gt_pose["obj_id"] = 1 # only 1 class

        scene_gt[key].append(gt_pose)

        import ipdb;ipdb.set_trace()
    
    scene_gt = dict(sorted(scene_gt.items()))
    print("Calculate pose ... done")

    # Write pose file
    # scene_gt_json = os.path.join(bop_test_pth, '000001', 'scene_gt.json')

    # with open(scene_gt_json, 'w') as f:
    #     json.dump(scene_gt, f, indent=4)

    # print("Generate pose file ... done")
