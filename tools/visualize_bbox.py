import cv2
import numpy as np

# image_path = "/data5/tangjw/underwater/underwater_dataset/underwater_bop/train/000001/rgb/45162.png"
# image_result_path = "/data4/fbw/deep_underwater_localization/data/output/45162_bbox.png"
# bbox = np.array([[445, 64], [571, 234]])
# corners2D = np.array([[505, 151], [519, 243], [546, 227], [555, 209], [586, 191], [440, 119], [466, 105], [458, 61], [489, 44]])

# image_path = "/data5/tangjw/underwater/underwater_dataset/underwater_bop/test/000001/rgb/0.png"
# image_result_path = "/data4/fbw/deep_underwater_localization/data/output/0_bbox.png"
# corners2D = np.array([[630, 278], [644, 237], [436, 287], [432, 249], [582, 111], [589, 68], [433, 125], [430, 85], [278, 644]])

image_path = "/data5/tangjw/underwater/underwater_dataset/underwater_bop/test/000001/rgb/002600.png"
image_result_path = "/data4/fbw/deep_underwater_localization/data/output/2600_bbox.png"
bbox = np.array([[223, 412], [394, 532]])
corners2D = np.array([[420, 513], [422, 468], [275, 555], [276, 503], [318, 435], [319, 400], [203, 456], [203, 418]])
image = cv2.imread(image_path)
cv2.rectangle(image, (bbox[0, 0],bbox[0, 1]), (bbox[1, 0], bbox[1, 1]), (255,0,0), 2)

for point in corners2D:
    cv2.circle(image, (point[0], point[1]), 1, (0, 0, 255), -1)

cv2.imwrite(image_result_path, image)