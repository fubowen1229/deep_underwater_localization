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
    train_file = args.train_file
    bop_root = '/data5/tangjw/underwater/underwater_dataset/underwater_bop'
    bop_train_pth = os.path.join(bop_root, 'train')

    # Acquire 2D points
    points_2d = {}
    with open(train_file, 'r') as f:        
        lines = f.readlines()
        for line in lines:
            info = line.strip().split(" ")
            image_id = info[0]
            points_2d[image_id] = {}

            points_2d[image_id]['image_pth']  = info[1]
            points_2d[image_id]['image_size'] = [int(i) for i in info[2:4]]
            points_2d[image_id]['label']      = int(info[4]) # always 0 (only 1 class)
            bbox                              = [int(i) for i in info[5:9]] # x_min y_min x_max y_max
            points_2d[image_id]['bbox']       = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]] # x y h w
            points_2d[image_id]['center_2d']  = [int(i) for i in info[9:11]] # different from test
            points_2d[image_id]['corner_2d']  = [int(i) for i in info[11:27]]

    # Calculate gt info
    scene_gt_info = {}
    for key in points_2d.keys():
        # Prepare for writing scene_gt_info.json
        # scene_gt[key] = []
        image_name = int(os.path.basename(points_2d[key]['image_pth'])[0:-4])
        scene_gt_info[image_name] = []
        gt_info = {}
        gt_info["bbox_obj"] = points_2d[key]['bbox']
        gt_info["bbox_visib"] = points_2d[key]['bbox']

        mask_image = os.path.join(bop_train_pth, '000001', 'mask', '{0:06d}_000000.png'.format(image_name))   
        mask = cv2.imread(mask_image)[:, :, 0]
        gt_info["px_count_all"] = np.count_nonzero(mask > 0)
        gt_info["px_count_valid"] = 0.0
        gt_info["px_count_visib"] = gt_info["px_count_all"]
        gt_info["visib_fract"] = 1.0

        scene_gt_info[image_name].append(gt_info)
    
    scene_gt_info = dict(sorted(scene_gt_info.items()))
    print("Calculate gt info done")

    # Write pose file
    scene_gt_json = os.path.join(bop_train_pth, '000001', 'scene_gt_info.json')

    with open(scene_gt_json, 'w') as f:
        json.dump(scene_gt_info, f, indent=4)

    print("Generate gt info json done")
