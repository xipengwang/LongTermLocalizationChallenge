#!/usr/bin/env python3

import argparse
import numpy as np
import os
import shutil
from tqdm import tqdm
import types
import cv2 as cv
import matplotlib.pyplot as plt

from reconstruction_pipeline import recover_database_images_and_ids

PLOT = False


def export_features(images, paths, args):
    # Export the features.
    print('Exporting features...')

    for image_name, _ in tqdm(images.items(), total=len(images.items())):
        image_path = os.path.join(paths.image_path, image_name)
        if os.path.exists(image_path):
            sift = cv.xfeatures2d.SIFT_create()
            img0 = cv.imread(image_path)
            gray = cv.cvtColor(img0, cv.COLOR_BGR2GRAY)
            keypoints, descs = sift.detectAndCompute(gray, None)
            kps = np.asarray([[kp.pt[0], kp.pt[1]] for kp in keypoints])
            img = cv.drawKeypoints(gray, keypoints, np.array([]), (0, 0, 255),
                                   cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            if PLOT:
                _, (ax1, ax2) = plt.subplots(2, 1)
                ax1.imshow(img)
                ax2.imshow(cv.cvtColor(img0, cv.COLOR_BGR2RGB))
                ax2.scatter(kps[:, 0], kps[:, 1])
                plt.show()
        features_path = os.path.join(paths.image_path, '%s.%s' % (image_name, args.method_name))
        np.savez(features_path, keypoints=kps, descriptors=descs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True, help='Path to the dataset')
    parser.add_argument('--method_name', required=True, help='Name of the method')
    args = parser.parse_args()

    # Create the extra paths.
    paths = types.SimpleNamespace()
    paths.database_path = os.path.join(args.dataset_path, 'database.db')
    # paths.database_path = os.path.join(args.dataset_path, 'aachen.db')
    paths.image_path = os.path.join(args.dataset_path, 'images', 'images_upright')

    images, _ = recover_database_images_and_ids(paths, args)
    export_features(images, paths, args)

    return


if __name__ == "__main__":
    main()
