import os
import glob
import tqdm

root_pth = '/data5/tangjw/underwater/underwater_dataset/underwater_bop/test/000001/rgb'
names = sorted(os.listdir(root_pth))
for name in names:
    src_pth = os.path.join(root_pth, name)
    file, extend = os.path.splitext(name)
    dst_pth = os.path.join(root_pth, '{:06d}.png'.format(int(file)))
    os.rename(src_pth, dst_pth)
    print(name)
    