import cv2
import numpy as np
import os
import json
from shutil import copyfile

import _init_paths
from meshply import MeshPly
from misc_utils import get_3D_corners, solve_pnp, get_camera_intrinsic

if __name__ == '__main__':
    # Path
    test_file = '/data4/fbw/deep_underwater_localization/data/my_data/pool_test.txt'
    test_dir_src= '/data4/fbw/deep_underwater_localization/dataset/pool/images'
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
            points_2d[image_id]['bbox'] = [int(i) for i in info[5:9]]
            points_2d[image_id]['center_2d'] = [int(i) for i in info[9:11]]
            points_2d[image_id]['corner_2d'] = [int(i) for i in info[11:27]]
    
    idx = 1
    for key, value in points_2d.items():
        image_src = os.path.join(test_dir_src, os.path.basename(points_2d[key]['image_pth']))
        image_dst = os.path.join(bop_test_pth, '000001', 'rgb', '{:06d}.png'.format(int(key)))
        im_src = cv2.imread(image_src)
        if int(key) < 9063:
            im_resize = cv2.resize(im_src, None, fx=0.5, fy=0.5)
        else:
            im_resize = im_src
        #import ipdb;ipdb.set_trace()
        assert im_resize.shape[0] == 600 # 600 * 800
        cv2.imwrite(image_dst, im_resize)
        print(f"{idx} / {len(points_2d)}")
        idx = idx + 1