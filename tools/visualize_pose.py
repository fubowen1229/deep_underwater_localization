import cv2
import numpy as np
import os
import json
from plyfile import PlyData

import _init_paths
from meshply import MeshPly

def get_3D_corners(vertices):
    min_x = np.min(vertices[0, :])
    max_x = np.max(vertices[0, :])
    min_y = np.min(vertices[1, :])
    max_y = np.max(vertices[1, :])
    min_z = np.min(vertices[2, :])
    max_z = np.max(vertices[2, :])

    corners = np.array([[min_x, min_y, min_z],
                        [min_x, min_y, max_z],
                        [min_x, max_y, min_z],
                        [min_x, max_y, max_z],
                        [max_x, min_y, min_z],
                        [max_x, min_y, max_z],
                        [max_x, max_y, min_z],
                        [max_x, max_y, max_z]])

    corners = np.concatenate((np.transpose(corners), np.ones((1, 8))), axis=0)
    return corners

def load_ply_model(model_path):
    ply = PlyData.read(model_path)
    data = ply.elements[0].data
    x = data["x"] * 1000.0
    y = data["y"] * 1000.0
    z = data["z"] * 1000.0
    return np.stack([x, y, z], axis=-1)

def project_model(model, pose, intrinsic_matrix):
    camera_points_3d_w = np.dot(model, pose[:, :3].T) + pose[:, 3]
    camera_points_3d_c = np.dot(camera_points_3d_w, intrinsic_matrix.T)
    return camera_points_3d_c[:, :2] / camera_points_3d_c[:, 2:], camera_points_3d_w[:, 2]

if __name__ == '__main__':
    # Path
    bop_root = '/data5/tangjw/underwater/underwater_dataset/underwater_bop'
    #bop_root = '/data/wanggu/Storage/BOP_DATASETS/ycbv'
    mesh_pth = os.path.join(bop_root, 'models', 'obj_000001_resample.ply')
    bop_train_pth = os.path.join(bop_root, 'train', '000001')
    bop_test_pth = os.path.join(bop_root, 'test', '000001')
    output_dir = '../data/output'

    # Acquire 3D corners
    mesh = MeshPly(mesh_pth)
    vertices = np.c_[np.array(mesh.vertices), np.ones((len(mesh.vertices), 1))].transpose()
    # corner_3d = get_3D_corners(vertices)
    # corner_3d = np.array([[-0.318758, -0.318758, -0.318758, -0.318758,  0.318758,  0.318758,    0.318758,  0.318758],
    #                     [-0.216993, -0.216993,  0.216965,  0.216965, -0.216993, -0.216993,   0.216965,  0.216965],
    #                     [-0.064505,  0.07    , -0.064505,  0.07    , -0.064505,  0.07    ,   -0.064505,  0.07    ],
    #                     [ 1.      ,  1.      ,  1.      ,  1.      ,  1.      ,  1.      ,   1.      ,  1.      ]])
    # corner_3d = corner_3d[0: 3, :].T.reshape(-1, 1, 3) # 8 * 1 * 3

    # Acquire camera matrix
    camera_intrinsic = np.array([[569.31671203, 0.0, 360.09063137],
	                            [0.0, 569.387306625, 301.45327471],
	                            [0.0, 0.0, 1.0]])

    # Read pose file
    scene_gt_json = os.path.join(bop_test_pth, 'scene_gt.json')
    with open(scene_gt_json, 'r') as f:
        scene_gt = json.load(f)

    # Visualize pose
    ply_model = load_ply_model(mesh_pth)
    for key in scene_gt.keys():
        if int(key) % 100 != 0:
            continue
        if int(key) > 10900:
            continue
        output_pth = os.path.join(output_dir, '{:06d}_pose.png'.format(int(key)))
        R = np.array(scene_gt[key][0]['cam_R_m2c']).reshape(3,3)
        T = np.array(scene_gt[key][0]['cam_t_m2c']).reshape(3,1)
        pose = np.concatenate((R, T), axis=1)
        points_project, z = project_model(ply_model, pose, camera_intrinsic)

        image_pth = os.path.join(bop_test_pth, 'rgb', '{:06d}.png'.format(int(key)))
        image = cv2.imread(image_pth, cv2.IMREAD_COLOR)
        # for i in range(points_project.shape[0]):
        #     x = int(round(np.array(points_project)[i][0]))
        #     y = int(round(np.array(points_project)[i][1]))
        #     image = cv2.circle(image, (x, y), 1, (255, 255, 255), thickness=-1)
        mask = np.zeros_like(image)
        try:
            for x, y in points_project:
                if x>image.shape[1] or x<0 or y<0 or y>image.shape[0]:
                    pass
                else:
                    mask[int(y), int(x), :] = 255
        except:
            pass
        image=((0.3*mask + 0.7*image).astype(np.uint8))
        
        cv2.imwrite(output_pth, image)
        print(f"Visualize {output_pth}")
        #import ipdb;ipdb.set_trace()
        #break

    print("Visualize pose ... done")
