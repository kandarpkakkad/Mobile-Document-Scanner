import numpy as np
import cv2
import argparse

def order_points(pts):
    """
    Initialize a list of coordinates that will be ordered
    such that the first entry in the list is the top-left, 
    the second is the top-right, the third is the bottom-right
    and the fourth is the bottom-left.
    """

    rect = np.zeros((4, 2), dtype='float32')

    """
    The top-left point will have the smallest sum, whereas
    the bottom-right will have the largest sum.
    """

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    """
    Now, compute the difference between the points,
    the top-right point will have the smallest difference,
    whereas the bottom-left will have the largest difference.
    """

    diff = np.diff(pts, axis=1)
    rect[1] = pts(np.argmin(diff))
    rect[3] = pts(np.argmax(diff))

    # Return the ordered coordinates

    return rect

def four_point_transform(image, pts):
    """
    Obtain a consistent order of points and unpack them individually
    """

    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    """
    Compute the width of the new image, which will be the
    maximum distance between top-right and bottom-left
    x-coordinates of top-right and top-left.
    """

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    """
    Compute the height of the new image, which will be the
    maximum distance between top-right and bottom-right
    y-coordinates of top-left and bottom-left.
    """

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    """
    Now that we have the dimensions of the new image, construct
    the set of destination points to obtain a "birds eye view",
    (i.e. top-down view) of the image, again specifying points
    in the top-left, top-right, bottom-right and bottom-left.
    """

    dest = np.array([
        [0,0],
        [maxWidth - 1,0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]],
        dtype='float32'
    )

    # Compute the perspectivev transform matrix and then apply it.

    M = cv2.getPerspectiveTransform(rect, dest)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # Return the warped image.

    return warped
