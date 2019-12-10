# detection.py
# Detection functions

import cv2
import numpy as np


def unpack_pt(pt):
    return tuple(pt[0].tolist())


def detect_rectangles(img, min_contour_area: int = 1000):
    """
    Detect and filter rectangles in an image
    :param img: Input image
    :param min_contour_area: Optional minimum contour area to detect, default = 1000 px^2
    :return: Detected rectangle contours
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img, 100, 200)
    img_clean = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8)))

    contours, hierarchy = cv2.findContours(img_clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    rect_contours = []
    rect_approx = []
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        contour_arc = cv2.arcLength(contour, True)
        if contour_area > min_contour_area:
            approx = cv2.approxPolyDP(contour, 0.02 * contour_arc, True)
            if len(approx) == 4:
                rect_contours.append(contour)
                rect_approx.append(approx)

    img_copy = img.copy()

    sorted_sets = []
    for unsorted_corners in rect_approx:
        labeled_corners = list(enumerate(unsorted_corners))
        x_sorted = sorted(labeled_corners, key=lambda pt: pt[1][0][0])
        y_sorted = sorted(labeled_corners, key=lambda pt: pt[1][0][1])
        x_ordering = [pt[0] for pt in x_sorted]
        y_ordering = [pt[0] for pt in y_sorted]
        upper_left = -1
        bottom_left = -1
        upper_right = -1
        bottom_right = -1
        for label_x in x_ordering[:2]:
            if label_x in y_ordering[:2]:
                upper_left = label_x
            else:
                bottom_left = label_x
        for label_y in y_ordering[:2]:
            if label_y != upper_left:
                upper_right = label_y
        for label_y in y_ordering[2:]:
            if label_y != bottom_left:
                bottom_right = label_y
        cv2.circle(img_copy, unpack_pt(unsorted_corners[upper_left]), 10, (0, 0, 0), 3)
        cv2.circle(img_copy, unpack_pt(unsorted_corners[upper_right]), 10, (255, 0, 0), 3)
        cv2.circle(img_copy, unpack_pt(unsorted_corners[bottom_left]), 10, (0, 255, 0), 3)
        cv2.circle(img_copy, unpack_pt(unsorted_corners[bottom_right]), 10, (0, 0, 255), 3)

        sorted_corners = [unpack_pt(unsorted_corners[pt]) for pt in (upper_left, upper_right, bottom_left, bottom_right)]
        sorted_sets.append(sorted_corners)

    # cv2.imshow('ordering test', img_copy)

    return rect_contours, sorted_sets
