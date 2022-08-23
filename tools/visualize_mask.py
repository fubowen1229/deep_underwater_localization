import cv2
import numpy as np
import os
import json

import _init_paths

if __name__ == '__main__':
    # Path
    bop_root = '/data5/tangjw/underwater/underwater_dataset/underwater_bop'
    #bop_root = '/data/wanggu/Storage/BOP_DATASETS/ycbv'
    bop_train_pth = os.path.join(bop_root, 'train', '000001')
    bop_test_pth = os.path.join(bop_root, 'test', '000001')
    output_dir = '../data/output/mask'
    os.makedirs(output_dir, exist_ok=True)

    # Read pose file
    scene_gt_json = os.path.join(bop_test_pth, 'scene_gt.json')
    with open(scene_gt_json, 'r') as f:
        scene_gt = json.load(f)

    # Visualize pose
    for key in scene_gt.keys():
        if int(key) % 100 != 0:
            continue
        if int(key) > 5000:
            continue
        output_pth = os.path.join(output_dir, '{:06d}.png'.format(int(key)))

        image_pth = os.path.join(bop_test_pth, 'rgb', '{:06d}.png'.format(int(key)))
        mask_pth = os.path.join(bop_test_pth, 'mask', '{:06d}_000000.png'.format(int(key)))
        image = cv2.imread(image_pth, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_pth, cv2.IMREAD_COLOR)

        image=((0.3*mask + 0.7*image).astype(np.uint8))
        
        cv2.imwrite(output_pth, image)
        print(f"Visualize {output_pth}")
        #import ipdb;ipdb.set_trace()
        #break

    print("Visualize pose ... done")
