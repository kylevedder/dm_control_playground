#!/usr/bin/env python3
import argparse
import glob
import joblib
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('dataset_folder', help='Folder to create datset in')
args = parser.parse_args()

def stat(pc):
  xmin, xmax = pc[:, 0].min(), pc[:, 0].max()
  ymin, ymax = pc[:, 1].min(), pc[:, 1].max()
  zmin, zmax = pc[:, 2].min(), pc[:, 2].max()
  return xmin, xmax, ymin, ymax, zmin, zmax

def stat_lst(lst):
  res = []
  for idx in range(3):
    res.append((min(lst[:, idx *2 ]), max(lst[:, idx*2 + 1])))
  return res

pc_stats = np.array([stat(joblib.load(pcf)) for pcf in glob.glob(args.dataset_folder + "/*")])
print(stat_lst(pc_stats))