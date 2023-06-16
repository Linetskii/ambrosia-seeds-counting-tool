"""Picture processing"""
from skimage import segmentation
from skimage import io
from skimage import measure
from skimage.feature import corner_peaks
from scipy import ndimage
import numpy as np
import imutils
import cv2 as cv


class Sample:
    """Stores and processes the seeds photo"""
    def __init__(self, path):
        self.image = io.imread(path)
        self.gray = self.__make_grayscale()
        self.thresh = self.__make_thresholds()

    @staticmethod
    def __correct_img(img, alpha=2, gamma=-80):
        """improves brightness and contrast"""
        bc = cv.addWeighted(img, alpha, img, 0, gamma)
        return bc

    def __make_grayscale(self):
        shifted = cv.pyrMeanShiftFiltering(self.image, 21, 51)
        corrected = self.__correct_img(shifted)
        gray = cv.cvtColor(corrected, cv.COLOR_BGR2GRAY)
        return cv.medianBlur(gray, 5)

    def __make_thresholds(self):
        thresh = cv.threshold(self.gray, 0, 255, cv.THRESH_BINARY_INV+cv.THRESH_OTSU, 11)[1]
        return thresh

    def __watershed(self):
        """Watershed algorythm implementation for seeds count"""
        d = ndimage.distance_transform_edt(self.thresh)
        local_max = corner_peaks(d, indices=False, min_distance=12, labels=self.thresh)
        markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
        labels = segmentation.watershed(-d, markers, mask=self.thresh)
        unique_labels = np.unique(labels)
        return labels, unique_labels

    @staticmethod
    def __is_close(labels, unique_labels):
        """Check, if area of segment is close enough to median area. Returns list of labels for big enough segments."""
        areas = np.array([(i.label, i.area) for i in measure.regionprops(labels)])
        median = np.median(np.array([i[1] for i in areas]))
        low = median // 1.5
        high = median * 1.5
        labels_true = []
        for i in areas:
            if i[1] > high:
                n_seeds = round(i[1] / median)
                for j in range(n_seeds):
                    labels_true.append(unique_labels[int(i[0])])
            if i[1] > low:
                labels_true.append(unique_labels[int(i[0])])
        return labels_true

    def __put_labels(self, labels, labels_true):
        """returns the labeled image"""
        img = self.image[:]
        for label in labels_true:
            # 0 for background
            if label == 0:
                continue
            # otherwise, allocate memory for the label region and draw it on the mask
            mask = np.zeros(self.gray.shape, dtype="uint8")
            mask[labels == label] = 255
            # detect contours in the mask and grab the largest one
            contours = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            c = max(contours, key=cv.contourArea)
            # draw a circle enclosing the object
            ((x, y), r) = cv.minEnclosingCircle(c)
            # cv.circle(img, (int(x), int(y)), int(r), (0, 255, 0), 1)
            cv.putText(img, "{}".format(label), (int(x) - 10, int(y)),
                       cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return segmentation.mark_boundaries(img, labels, color=(1, 0, 0))

    def count(self):
        """count and label the ambrosia seeds"""
        labels, unique_labels = self.__watershed()
        labels_true = self.__is_close(labels, unique_labels)
        print(f"[INFO] {len(labels_true) - 1} seeds found by watershed algorythm")
        labeled_img = self.__put_labels(labels, labels_true)
        return labeled_img, len(labels_true) - 1
