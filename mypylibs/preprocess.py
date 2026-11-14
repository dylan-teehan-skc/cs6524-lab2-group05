import sys

import cv2 as cv

LONG, SHORT, FPS = 640, 360, 30


def preprocess(src, dst):
    cap = cv.VideoCapture(src)
    if not cap.isOpened():
        raise FileNotFoundError(src)
    w, h = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    size = (LONG, SHORT) if w >= h else (SHORT, LONG)  # keep orientation, don't rotate
    # H.264 (avc1): mp4v made a 6.5MB file, over the 5MB limit
    out = cv.VideoWriter(dst, cv.VideoWriter_fourcc(*'avc1'), FPS, size)
    step = max(1, round(cap.get(cv.CAP_PROP_FPS) / FPS))  # 60 fps source -> keep every 2nd frame
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % step == 0:
            out.write(cv.resize(frame, size, interpolation=cv.INTER_AREA))
        n += 1
    cap.release()
    out.release()
    print(f'{src} ({w}x{h}) -> {dst} ({size[0]}x{size[1]} @ {FPS} fps, {(n + step - 1) // step} frames)')


if __name__ == '__main__':
    preprocess(sys.argv[1], sys.argv[2])
