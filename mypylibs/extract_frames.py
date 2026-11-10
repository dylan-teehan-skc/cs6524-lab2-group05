import os
import sys

import cv2 as cv

N = 30

video, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
cap = cv.VideoCapture(video)
total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
for k in range(N):
    cap.set(cv.CAP_PROP_POS_FRAMES, k * total // N)
    ok, img = cap.read()
    if ok:
        cv.imwrite(os.path.join(out_dir, f'{k:02}.jpg'), img)
