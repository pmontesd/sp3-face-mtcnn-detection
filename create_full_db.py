from shutil import copyfile
from pathlib import Path
import sys
import argparse

def get_new_path(curr_path, orig_path, dest_path):
    relative = curr_path.relative_to(orig_path)
    return dest_path.joinpath(relative)

def is_target_missing(target):
    return not target.parent.is_dir()

def create_target_dir(target):
    target.parent.mkdir(parents=True)

parser = argparse.ArgumentParser()
parser.add_argument("ref_dir")
parser.add_argument("src_dir")
parser.add_argument("dst_dir")
args = parser.parse_args()

r = Path(args.ref_dir)
s = Path(args.src_dir)
d = Path(args.dst_dir)

for imageFile in list(r.glob('**/*.jpg')):
    source = get_new_path(imageFile, r, s)
    target = get_new_path(imageFile, r, d)
    if is_target_missing(target):
        create_target_dir(target)
    copyfile(source, target)
