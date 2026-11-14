import cv2 as cv
import matplotlib
matplotlib.use('Agg')  # no screen in CI
import matplotlib.pyplot as plt
import numpy as np

MAX_DEPTH = 100  # colour scale and point cloud range

sgbm = cv.StereoSGBM_create(numDisparities=64, blockSize=7, P1=8 * 49, P2=32 * 49,
                            uniquenessRatio=10, speckleWindowSize=100, speckleRange=2)


def rectify(img1, img2, K, D, R, t):
    size = img1.shape[1::-1]
    R1, R2, P1, P2, *_ = cv.stereoRectify(K, D, K, D, size, R, t)
    map1 = cv.initUndistortRectifyMap(K, D, R1, P1, size, cv.CV_32FC1)
    map2 = cv.initUndistortRectifyMap(K, D, R2, P2, size, cv.CV_32FC1)
    left = cv.remap(img1, *map1, cv.INTER_LINEAR)
    right = cv.remap(img2, *map2, cv.INTER_LINEAR)
    if t[0, 0] > 0:  # camera moved left, so the second frame is the left view
        left, right = right, left
    return left, right, P1


def depth_map(left, right, f, B):
    disparity = sgbm.compute(left, right) / 16  # SGBM returns disparity * 16
    disparity[disparity <= 0] = np.nan          # no match
    return f * B / disparity


def write_video(depths, path):
    fig, ax = plt.subplots(figsize=(4, 6), dpi=100)
    im = ax.imshow(depths[0], cmap='turbo', vmin=0, vmax=MAX_DEPTH)
    fig.colorbar(im, label='depth (cm)')
    ax.axis('off')

    out = None
    for depth in depths:
        im.set_data(depth)
        fig.canvas.draw()
        frame = cv.cvtColor(np.asarray(fig.canvas.buffer_rgba()), cv.COLOR_RGBA2BGR)
        if out is None:
            out = cv.VideoWriter(path, cv.VideoWriter_fourcc(*'mp4v'), 5, frame.shape[1::-1])
        out.write(frame)
    out.release()
    plt.close(fig)


def write_ply(depth, image, P, path):
    # NaN compares False so unmatched pixels drop out; black = empty border left by rectification
    v, u = np.nonzero((depth < MAX_DEPTH) & image.any(axis=2))
    z = depth[v, u]
    x = (u - P[0, 2]) * z / P[0, 0]
    y = (v - P[1, 2]) * z / P[1, 1]
    rgb = image[v, u, ::-1]  # BGR -> RGB
    header = (f'ply\nformat ascii 1.0\nelement vertex {len(z)}\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header')
    # flip y and z so the cloud opens upright in CloudCompare
    np.savetxt(path, np.column_stack([x, -y, -z, rgb]), fmt='%.2f %.2f %.2f %d %d %d',
               header=header, comments='')
