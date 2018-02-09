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

        self.noise_kernel_3 = np.ones((3, 3), np.uint8)
        self.noise_kernel_2 = np.ones((2, 2), np.uint8)

        # grayscale for analysis
        self.gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # gaussian blur to smooth out features
        self.gauss = cv2.GaussianBlur(self.gray, (5, 5), 0)

        self.thresh = None
        self.method = None
        self.mask = None

    def get_thresh(self, method='thresh', thresh=125):
        """
        Gets the threshed image

        :param method: method to use
        :return:
        """
        methods = ['otsu', 'gauss', 'mean', 'thresh']
        if method not in methods:
            raise AttributeError('Incorred method selected: {}. Supported methods are: {}'.format(method, ', '.join(methods)))

        if method == 'thresh':
            _, threshed = cv2.threshold(self.gauss, thresh, 255, cv2.THRESH_BINARY_INV)
        if method == 'otsu':
            _, threshed = cv2.threshold(self.gauss, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if method == 'gauss':
            threshed = cv2.adaptiveThreshold(self.gauss, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        if method == 'mean':
            threshed = cv2.adaptiveThreshold(self.gauss, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # clean up noise
        morph = cv2.morphologyEx(threshed, cv2.MORPH_OPEN, self.noise_kernel_2, iterations=1)

        self.thresh = threshed

        return threshed

    def get_degen_mask(self, method, thresh=125):
        """
        Gets the mask of suspected degenerated axons

        :param method: method to use
        :return:
        """
        method_map = {
            'dt': self.get_degen_dt(thresh)
        }
        assert method in method_map.keys()

        return method_map[method]

    def get_degen_dt(self, thresh=125):
        """
        Gets mask using distance transform

        :param thresh: thresholding cutoff
        :return:
        """
        threshed = self.get_thresh(method='thresh', thresh=thresh)

        # do distance transform
        dist_transform = cv2.distanceTransform(threshed, cv2.DIST_L2, 5)

        # threshold the distance transform
        cutoff = dist_transform.max() ** 0.5
        _, sure_fg = cv2.threshold(dist_transform, cutoff, 255, 0)

        # clean up noise again
        sure_fg_morph = cv2.morphologyEx(sure_fg, cv2.MORPH_OPEN, self.noise_kernel_3, iterations=2)
        sure_fg_morph = cv2.convertScaleAbs(sure_fg_morph)

        # _, markers = cv2.connectedComponents(sure_fg_morph)

        self.method = dist_transform
        self.mask = sure_fg_morph

        return sure_fg_morph

    def get_centroids(self, mask, min_dist=0):
        """
        Calculates the center points of masked areas, and optionally throws out points
        that are too close to another point

        :param mask: mask suspected degenerated axons
        :param min_dist: min distance between points, 0 is don't throw any out
        :return:
        """
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        moments = list(map(cv2.moments, contours))

        cx = lambda m: int(m['m10'] / m['m00'])
        cy = lambda m: int(m['m01'] / m['m00'])

        centroids = [(cx(m), cy(m)) for m in moments]

        centroids = np.array(centroids)

        # TODO: factor out culling
        # this could be a problem if there is a long chain of points near each
        # other, we wouldn't want to throw them all out
        # maybe throw out single point, then do again (recurse)
        # potentially TODO: make recursive version which throws single point out at time
        if min_dist > 0 and len(centroids):
            c = centroids.copy()

            d = sp.spatial.distance.pdist(c, 'euclidean')
            e = sp.spatial.distance.squareform(d)
            e[e == 0] = np.NaN

            too_close = np.argwhere(e < min_dist)[:, 0]
            too_close = np.unique(too_close)
            too_close[::-1].sort()

            for i in too_close:
                c = np.delete(c, i, axis=0)

            centroids = c

        return centroids
