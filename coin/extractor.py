#!/usr/bin/env python
import os
import sys

import numpy as np
from cv2 import imwrite
import coin.segmenter
import coin.utils
from matlab import engine


FAKE_DATA = [
    np.eye(5), np.eye(5),
]

ALLOWED_FILES = ('.jpg', '.jpeg', '.png')


def get_files(input_directory):
    image_files = filter(
        lambda x: os.path.splitext(x)[1].lower() in ALLOWED_FILES,
        os.listdir(input_directory)
    )
    return image_files


def process_image(image_file_name, input_directory, output_directory, eng):
    image_file_path = os.path.join(input_directory, image_file_name)
    # Do something
    # Get a list of images
    segmented_images = coin.segmenter.processImg(image_file_path, eng)
    for idx, segmented_image in enumerate(segmented_images):
        image_name = "{0}-{1}.jpg".format(
            os.path.splitext(image_file_name)[0],
            idx,
        )
        write_image(output_directory, image_name, segmented_image)


def write_image(output_directory, image_name, image):
    imwrite(os.path.join(output_directory, image_name), image)


def main():
    if len(sys.argv) != 3:
        print "Usage: ./coin-segment-extract input_directory output_directory"
        sys.exit(1)
    input_directory = sys.argv[1]
    output_directory = sys.argv[2]
    if not (os.path.isdir(input_directory) and os.path.isdir(output_directory)):
        print "input_directory and/or output_directory are not directories!"
        sys.exit(1)

    eng = engine.start_matlab()
    coin.utils.load_matlab_util(eng)

    image_files = get_files(input_directory)
    for image_file in image_files:
        process_image(image_file, input_directory, output_directory, eng)

    sys.exit(0)


if __name__ == '__main__':
    main()
