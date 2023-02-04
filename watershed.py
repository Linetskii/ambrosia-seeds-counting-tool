# import the necessary packages
import skimage.segmentation
from skimage.feature import peak_local_max
from skimage import segmentation
from skimage import io
from skimage import color
from skimage import filters
from skimage import feature
from skimage import measure
from skimage import morphology
from skimage import exposure
from scipy import ndimage
import numpy as np
import imutils
import cv2 as cv

class Sample:
    def __init__(self, path):
        self.kernel = np.ones((3, 3), np.uint8)
        self.image = io.imread(path)
        self.gray, self.thresh = self.prepare_img()

    @staticmethod
    def bright_contr(img):
        alpha=2
        gamma=-80
        bc = cv.addWeighted(img, alpha, img, 0, gamma)
        return bc

    # def prepare_img2(self):
    #     img  = cv.pyrMeanShiftFiltering(self.image, 21, 51)
    #     img = self.bright_contr(self.image)
    #     gray = color.rgb2gray(img)
    #     gray = filters.gaussian(gray)
    #     thresh = filters.threshold_otsu(gray)
    #     thresh = gray > thresh
    #     return gray, thresh

    def prepare_img(self):
        shifted = cv.pyrMeanShiftFiltering(self.image, 21, 51)
        shifted = self.bright_contr(shifted)
        gray = cv.cvtColor(shifted, cv.COLOR_BGR2GRAY)
        gray = cv.medianBlur(gray, 5)
        thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU, 11)[1]
        # opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, self.kernel, iterations=2)
        # sure_bg = cv.dilate(opening, self.kernel, iterations=3)
        # dist_transform = cv.distanceTransform(thresh, cv.DIST_L2, 0)
        # ret, sure_fg = cv.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        # sure_fg = np.uint8(sure_fg)
        # unknown = cv.subtract(sure_bg, sure_fg)
        return gray, thresh

    def show(self, name):
        io.imshow( self.__dict__[name], plugin='matplotlib')
        io.show()

    def morph_gac(self):
        img = segmentation.morphsnakes.inverse_gaussian_gradient(self.gray, alpha=100, sigma=0.35)
        io.imshow(img)
        io.show()
        labels = segmentation.morphological_geodesic_active_contour(img, num_iter=50, balloon=5)
        labels_true = np.unique(labels)
        self.put_labels(labels, labels_true)

    def slic(self):
        labels = segmentation.slic(self.image)
        labels_true = np.unique(labels)
        self.put_labels(labels, labels_true)

    def random_walker(self):
        D = ndimage.distance_transform_edt(self.thresh)
        localMax = peak_local_max(D, indices=False, min_distance=12, labels=self.thresh)
        # io.imshow(localMax)
        # io.show()
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        print('get')
        labels = segmentation.random_walker(self.gray, markers)
        labels_true = np.unique(labels)
        print('ready')
        self.put_labels(labels, labels_true)

    def quickshift(self):
        labels = segmentation.quickshift(self.image, convert2lab=True)
        labels_true = np.unique(labels)
        self.put_labels(labels, labels_true)

    def watershed(self):
        D = ndimage.distance_transform_edt(self.thresh)
        localMax = peak_local_max(D, indices=False, min_distance=12, labels=self.thresh)
        # io.imshow(localMax)
        # io.show()
        markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
        # offset = morphology.disk(7)
        labels = segmentation.watershed(-D, markers, mask=self.thresh)
        unique_l = np.unique(labels)
        areas = np.array([(i.label, i.area) for i in measure.regionprops(labels)])
        median = np.median(np.array([i[1] for i in areas]))
        print(median)
        low = median // 1.5
        high = median * 1.5
        labels_true = []
        for i in areas:
            if i[1] > high:
                n_seeds = round(i[1]/median)
                print(n_seeds, i[1], i[0])
                for j in range(n_seeds):
                    labels_true.append(unique_l[i[0]])
            if i[1] > low:
                labels_true.append(unique_l[i[0]])
        print(type(labels_true[0]), labels_true)
        print("[INFO] {} unique segments found".format(len(labels_true) - 1))
        self.put_labels(labels, labels_true)

    def put_labels(self, labels, labels_true):
        print('put')
        img = self.image[:]
        for label in labels_true:
            # if the label is zero, we are examining the 'background'
            # so simply ignore it
            if label == 0:
                continue
            # otherwise, allocate memory for the label region and draw
            # it on the mask
            mask = np.zeros(self.gray.shape, dtype="uint8")
            mask[labels == label] = 255
            # detect contours in the mask and grab the largest one
            cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv.contourArea)
            # draw a circle enclosing the object
            ((x, y), r) = cv.minEnclosingCircle(c)
            # cv.circle(img, (int(x), int(y)), int(r), (0, 255, 0), 1)
            cv.putText(img, "{}".format(label), (int(x) - 10, int(y)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        io.imshow(skimage.segmentation.mark_boundaries(img, labels), plugin='matplotlib')
        io.show()

    # def find_features(self):
    #     descriptor_extractor = feature.ORB()
    #     descriptor_extractor.detect_and_extract(self.gray)