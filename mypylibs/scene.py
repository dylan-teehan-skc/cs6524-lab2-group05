import glob
from os import path as opath

import cv2 as cv
import numpy as np

from mypylibs.camera_motion import match_keypoints, relative_pose
from mypylibs.depth import MAX_DEPTH, depth_map, rectify, write_ply, write_video

GAP = 5  # neighbouring frames barely move, so pair each frame with the one 5 ahead


def process_scene(frames_dir, K, D):
    out = 'out/' + opath.basename(frames_dir).removesuffix('_frames').replace('_', '-')  # out/01-01
    frames = [cv.imread(p) for p in sorted(glob.glob(f'{frames_dir}/*.jpg'))]

    clouds = []
    for img1, img2 in zip(frames, frames[GAP:]):
        R, t = relative_pose(*match_keypoints(img1, img2), K)
        left, right, P = rectify(img1, img2, K, D, R, t)
        depth = depth_map(left, right, f=P[0, 0], B=np.linalg.norm(t))
        clouds.append((depth, left, P))

    write_video([depth for depth, _, _ in clouds], f'{out}.mp4')

    # point cloud from the pair with the most depth pixels in range
    best = max(clouds, key=lambda cloud: np.sum(cloud[0] < MAX_DEPTH))
    write_ply(*best, f'{out}.ply')
