"""Prelim image processing"""

from pathlib import Path

from matplotlib import pyplot as plt
import matplotlib as mpl

import cv2
import numpy as np
import scipy as sp

from scipy import spatial

__author__ = 'Alexander Tomlinson'
__email__ = 'tomlinsa@ohsu.edu'
__version__ = '0.0.1'


class ImageProcess:
    """
    Handles image processing
    """
    def __init__(self, im):
        """
        preprocesses image

        :param im: np array of image
        """
        self.im = im
        self.method_im = None
        self.noise_kernel_3 = np.ones((3, 3), np.uint8)
        self.noise_kernel_2 = np.ones((2, 2), np.uint8)

        # grayscale for analysis
        self.gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # gaussian blur to smooth out features
        self.gauss = cv2.GaussianBlur(self.gray, (5, 5), 0)

    def get_degen_mask(self, method):
        """
        Gets the mask of suspected degenerated axons

        :param method: method to use
        :return:
        """
        method_map = {
            'dt': self.get_degen_dt
        }
        assert method in method_map.keys()

        return method_map[method]()

    def get_degen_dt(self, fx=0.4):
        """
        Gets mask using distance transform

        :param fx:
        :return:
        """
        # use otsu to thresh
        _, otsu = cv2.threshold(self.gauss, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # clean up noise
        morph = cv2.morphologyEx(otsu, cv2.MORPH_OPEN, self.noise_kernel_2, iterations=1)

        # do distance transform
        dist_transform = cv2.distanceTransform(morph, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, fx * dist_transform.max(), 255, 0)

        sure_fg_morph = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, self.noise_kernel_3, iterations=2)
        sure_fg_morph = cv2.convertScaleAbs(sure_fg_morph)

        _, markers = cv2.connectedComponents(sure_fg_morph)
        #     print(markers.max())

        self.method_im = dist_transform

        return sure_fg_morph

    # TODO: decide if factor out culling?
    def get_centroids(self, mask, min_dist=0):
        """
        Calculates the center points of masked areas

        :param mask: mask suspected degenerated axons
        :param cull: min distance between points, 0 is don't throw any out
        :return:
        """
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        moments = list(map(cv2.moments, contours))

        cx = lambda m: int(m['m10'] / m['m00'])
        cy = lambda m: int(m['m01'] / m['m00'])

        centroids = [(cx(m), cy(m)) for m in moments]

        centroids = np.array(centroids)

        if min_dist > 0:
            c = centroids.copy()

            d = sp.spatial.distance.pdist(c, 'euclidean')
            e = sp.spatial.distance.squareform(d)
            e[e == 0] = np.NaN

            too_close = np.argwhere(e < min_dist)[:, 0]
            too_close[::-1].sort()

            for i in too_close:
                c = np.delete(c, i, axis=0)

            centroids = c

        return centroids
