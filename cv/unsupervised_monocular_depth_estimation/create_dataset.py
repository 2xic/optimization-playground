import sys
import argparse

import cv2
print(cv2.__version__)


def get_frames(path_in, path_out):
    count = 0
    video_capture = cv2.VideoCapture(path_in)
    success, image = video_capture.read()
    success = True
    while success:
        video_capture.set(cv2.CAP_PROP_POS_MSEC, (count * 1000 / 20))
        success, image = video_capture.read()
        print('Read a new frame: ', success)
        if not success:
            count += 1
          #  success = True
            continue
        cv2.imwrite(f"{path_out}/frame{count}.jpg", image)
        count += 1


if __name__ == "__main__":
    # uses the comma.ai dataset challenge dataset
    # https://github.com/commaai/speedchallenge
    # get_frames("./speedchallenge-master/data/train.mp4", "./dataset/")
    pass