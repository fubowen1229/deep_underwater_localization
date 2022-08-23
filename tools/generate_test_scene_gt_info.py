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
            points_2d[image_id]['corner_2d'] = [int(i) for i in info[5:21]] # different from train, no bbox & center
    
    # Calculate gt info
    scene_gt_info = {}
    for key in points_2d.keys():
        key = int(key)
        mask_image = os.path.join(bop_test_pth, '000001', 'mask', '{0:06d}_000000.png'.format(key))
        mask = cv2.imread(mask_image)[:, :, 0] / 255
        maskx = np.any(mask, axis=0)
        masky = np.any(mask, axis=1)
        x1 = np.argmax(maskx)
        y1 = np.argmax(masky)
        x2 = len(maskx) - np.argmax(maskx[::-1])
        y2 = len(masky) - np.argmax(masky[::-1])
        bbox = [int(x1), int(y1), int(x2) - int(x1), int(y2) - int(y1)] # x, y, h, w

        px_count_all = np.count_nonzero(mask > 0)

        scene_gt_info[key] = []
        gt_info = {}
        gt_info["bbox_obj"] = bbox
        gt_info["bbox_visib"] = bbox
        gt_info["px_count_all"] = px_count_all
        gt_info["px_count_valid"] = 0.0
        gt_info["px_count_visib"] = px_count_all
        gt_info["visib_fract"] = 1.0
        
        scene_gt_info[key].append(gt_info)

    scene_gt_info = dict(sorted(scene_gt_info.items()))
    print("Calculate gt info done")

     # Write pose file
    scene_gt_json = os.path.join(bop_test_pth, '000001', 'scene_gt_info.json')

    with open(scene_gt_json, 'w') as f:
        json.dump(scene_gt_info, f, indent=4)

    print("Generate gt info json done")