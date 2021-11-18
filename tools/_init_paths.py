import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add to PYTHONPATH
core_path = osp.join(this_dir, '..', 'core')
utils_path = osp.join(this_dir, '..', 'utils')
misc_path = osp.join(this_dir, '..', 'misc')
add_path(core_path)
add_path(utils_path)
add_path(misc_path)