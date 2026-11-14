import numpy as np
import cv2 as cv
import glob


def calibrate(cam: int):

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((5*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:5].T.reshape(-1,2) * 2.5

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob(f'in/{cam:02}_frames/*.jpg')
    if not images:
        raise FileNotFoundError(f'in/{cam:02}_frames')

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (8,5), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    np.savetxt(f'out/{cam:02}_K.csv', mtx, delimiter=',')
    np.savetxt(f'out/{cam:02}_D.csv', dist, delimiter=',')
    print(f'camera {cam:02} reprojection error: {ret:.3f} px')

    return mtx, dist
