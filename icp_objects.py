#!/usr/bin/env python3

import os
# Without this set, the multiprocessing API hangs during ICP
os.environ["OMP_NUM_THREADS"] = "1"

from matplotlib.pyplot import draw
from numpy.lib.utils import source
import joblib
import open3d as o3d
import numpy as np
import copy
import argparse
import glob
import multiprocessing
import functools


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def pc_to_np(pc):
    return np.asarray(pc.points)


def np_to_pc(arr):
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(arr))


def extract_object_by_id(pc, masks, id):
    return np_to_pc(np.asarray(pc.points)[masks == id])


def extract_background_np(pc, masks):
    return np.asarray(pc.points)[masks < 0]


def extract_background(pc, masks):
    return np_to_pc(extract_background(pc, masks))


def transform_pcs(input_dataset_folder, source_idx, target_idx):
    source_np = joblib.load(f"{input_dataset_folder}/{source_idx:06d}_pc.pkl")
    source_pc = np_to_pc(source_np)
    source_masks = joblib.load(
        f"{input_dataset_folder}/{source_idx:06d}_classes.pkl")

    target_np = joblib.load(f"{input_dataset_folder}/{target_idx:06d}_pc.pkl")
    target_pc = np_to_pc(target_np)
    target_masks = joblib.load(
        f"{input_dataset_folder}/{target_idx:06d}_classes.pkl")

    object_ids = np.intersect1d(np.unique(source_masks),
                                np.unique(target_masks))
    # Remove background classes by filtering negative ids
    object_ids = object_ids[object_ids >= 0]

    def get_transformed_target(source_pc, target_pc):
        threshold = 0.01
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_pc, target_pc, threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        target_pc.transform(np.linalg.inv(reg_p2p.transformation))
        return target_pc

    source_pcs = [
        extract_object_by_id(source_pc, source_masks, oid)
        for oid in object_ids
    ]
    target_pcs = [
        extract_object_by_id(target_pc, target_masks, oid)
        for oid in object_ids
    ]
    transformed_target_pcs = [
        get_transformed_target(spc, tpc)
        for spc, tpc in zip(source_pcs, target_pcs)
    ]
    source_bg_pc_pts = extract_background_np(source_pc, source_masks)
    transformed_target_pc_with_source_bg = np_to_pc(
        np.concatenate([source_bg_pc_pts] +
                       [pc_to_np(pc) for pc in transformed_target_pcs]))
    return source_pc, transformed_target_pc_with_source_bg


def process_idx(input_dataset_folder, output_dataset_folder, idx):
    print(f"Processing pair {idx:06d} - {idx + 1:06d}")
    source_pc, transformed_target_pc = transform_pcs(input_dataset_folder, idx,
                                                     idx + 1)
    joblib.dump(pc_to_np(source_pc),
                f"{output_dataset_folder}/input/{idx:06}.pkl")
    joblib.dump(pc_to_np(transformed_target_pc),
                f"{output_dataset_folder}/pair/{idx:06}.pkl")


parser = argparse.ArgumentParser()
parser.add_argument('input_dataset_folder',
                    help='Folder with the created dataset')
parser.add_argument('output_dataset_folder',
                    help='Folder with the created dataset')
parser.add_argument('--num_cpus',
                    help='Number of CPUs to use',
                    type=int,
                    default=multiprocessing.cpu_count(),
                    required=False)
args = parser.parse_args()

input_dataset_folder = args.input_dataset_folder
output_dataset_folder = args.output_dataset_folder

from pathlib import Path

Path(f"{output_dataset_folder}/input").mkdir(parents=True, exist_ok=True)
Path(f"{output_dataset_folder}/pair").mkdir(parents=True, exist_ok=True)

source_files = sorted(list(glob.glob(f"{input_dataset_folder}/*_pc.pkl")))

assert (len(source_files) > 1)

with multiprocessing.Pool(args.num_cpus) as p:
    p.map(
        functools.partial(process_idx, input_dataset_folder,
                          output_dataset_folder), range(len(source_files) - 1))
