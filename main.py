import glob

from mypylibs.calibration import calibrate
from mypylibs.scene import process_scene


def main(cam: int):

    K, D = calibrate(cam)

    for frames_dir in sorted(glob.glob(f'in/{cam:02}_[0-9][0-9]_frames')):
        process_scene(frames_dir, K, D)


if __name__ == '__main__':

    cam = 0
    while(True):
        cam += 1
        try:
            main(cam)
        except FileNotFoundError:
            break
