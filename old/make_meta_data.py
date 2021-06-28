#!/usr/bin/env python

import glob
import sys
import numpy as np

dir_root = sys.argv[1]

def make_info_file(folder, fname):
  files = sorted(list(glob.glob(dir_root + "/" + folder + "/*")))
  print(files)
  names = [e.split("/")[-1].split('.')[0] + '\n' for e in files]
  f = open(dir_root + "/ImageSets/" + fname, 'w')
  f.writelines(names)
  f.close()

def make_fake_img(folder):
  files = sorted(list(glob.glob(dir_root + "/" + folder + "/label_2/*")))
  print(files)
  names = sorted([e.split("/")[-1].split('.')[0] for e in files])
  for n in names:
    pth = dir_root + '/' + folder + "/image_2/{}.png".format(n)
    img = Image.new('RGB', (1,1), color='white')
    img.save(pth)

def identity_proj():
  arr = np.zeros((3,4))
  arr[0,0] = 1
  arr[1,1] = 1
  arr[2,2] = 1
  return arr

def write_arr(f, arr):
  for r in arr:    
    f.write(' '.join([str(e) for e in r]) + ' ')
  f.write('\n')

def make_fake_calib(folder):
  files = sorted(list(glob.glob(dir_root + "/" + folder + "/label_2/*")))
  print(files)
  names = sorted([e.split("/")[-1].split('.')[0] for e in files])
  for n in names:
    pth = dir_root + '/' + folder + "/calib/{}.txt".format(n)
    f = open(pth, 'w')
    f.write("P0: ")
    write_arr(f, identity_proj())
    f.write("P1: ")
    write_arr(f, identity_proj())
    f.write("P2: ")
    write_arr(f, identity_proj())
    f.write("P3: ")
    write_arr(f, identity_proj())
    f.write("R0_rect: ")
    write_arr(f, np.eye(3))
    f.write("Tr_velo_to_cam: ")
    write_arr(f, identity_proj())
    f.write("Tr_imu_to_velo: ")
    write_arr(f, identity_proj())
  
from pathlib import Path
Path(dir_root + "/ImageSets/").mkdir(parents=True, exist_ok=True)
make_info_file("training/label_2", "train.txt")
make_info_file("testing/label_2", "test.txt")
make_info_file("testing/label_2", "val.txt")


# Path(dir_root + "/training/image_2/").mkdir(parents=True, exist_ok=True)
# Path(dir_root + "/testing/image_2/").mkdir(parents=True, exist_ok=True)

# from PIL import Image

# make_fake_img("training/")
# make_fake_img("testing/")

# Path(dir_root + "/training/calib/").mkdir(parents=True, exist_ok=True)
# Path(dir_root + "/testing/calib/").mkdir(parents=True, exist_ok=True)

# make_fake_calib("training/")
# make_fake_calib("testing/")