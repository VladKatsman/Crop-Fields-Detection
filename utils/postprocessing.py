import cv2
import numpy as np
import torch
from google_maps_python.map_utils import LatLngTranslator

# tmp
from PIL import Image


def denoise(mask, eps):
    """Removes noise from a mask.
    Args:
      mask: the mask to remove noise from.
      eps: the morphological operation's kernel size for noise removal, in pixel.
    Returns:
      The mask after applying denoising.
    """

    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eps, eps))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, struct)


def grow(mask, eps):
    """Grows a mask to fill in small holes, e.g. to establish connectivity.
    Args:
      mask: the mask to grow.
      eps: the morphological operation's kernel size for growing, in pixel.
    Returns:
      The mask after filling in small holes.
    """

    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eps, eps))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, struct)


def erode(mask, eps):
    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eps, eps))
    return cv2.morphologyEx(mask, cv2.MORPH_ERODE, struct)


def dilate(mask, eps):
    struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (eps, eps))
    return cv2.morphologyEx(mask, cv2.MORPH_DILATE, struct)


def simplify(polygon, eps):
    """Simplifies a polygon to minimize the polygon's vertices.
    Args:
      polygon: the polygon made up of a list of vertices.
      eps: the approximation accuracy as max. percentage of the arc length, in [0, 1]
    """

    assert 0 <= eps <= 1, "approximation accuracy is percentage in [0, 1]"

    epsilon = eps * cv2.arcLength(polygon, closed=True)
    return cv2.approxPolyDP(polygon, epsilon=epsilon, closed=True)


def get_contours(mask):
    """Extracts contours and the relationship between them from a binary mask.
    Args:
      mask: the binary mask to find contours in.
    Returns:
      The detected contours as a list of points and the contour hierarchy.
    Note: the hierarchy can be used to re-construct polygons with holes as one entity.
    """

    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


class BoundaryHandler:
    """ Denoize and separation of boundaries using erosion and dilation"""

    def __init__(self, center, simplify_thresh=0.01, zoom=18):
        self.simplify_thresh = simplify_thresh
        self.translator = LatLngTranslator(center[0], center[1], zoom=zoom)

    def get_coordinates(self, mask, h_cor, w_cor):
        """ Translates pixel points of specific sub image of the final grid to the coordinates

            Arguments:
                mask (torch tensor): binary postprocessed result of the model prediction
                h_cor, w_cor (int): anchors for each sub grid in order to find accurate pix values
                                    with regard to the rectangle area center
        """
        mask = erode(mask, 5)
        contours = get_contours(mask)
        contours = [simplify(c, self.simplify_thresh) for c in contours if cv2.contourArea(c) > 700]

        farm_polygons = []
        for polygon in contours:
            polygon = [list(self.translator.local_pix2latlng(p[0][0] + h_cor, p[0][1] + w_cor)) for p in polygon]
            # repeat
            polygon.append(polygon[-1])
            farm_polygons.append(polygon)
        return farm_polygons


def process_mask(mask):
    """ Returns postprocessed masks and list of polygons of the farms"""

    # create binary masks to find contours
    mask[mask == 2] = 0
    mask = grow(mask, 5)
    filler = np.zeros_like(mask)
    contours = get_contours(mask)
    if len(contours) < 1:
        return filler
    polygons = []
    for contour in contours:
        tmp_mask = np.zeros_like(mask)
        cv2.fillPoly(tmp_mask, [contour], 1)
        coef = 30
        separator = erode(tmp_mask, coef)
        tmp_contours = get_contours(separator)
        # remove noize
        tmp_contours = [c for c in tmp_contours if (cv2.contourArea(c) > 120)]
        if len(tmp_contours) > 1:
            # possibility to have 2 farms instead of 1
            tmp_mask_new = np.zeros_like(mask)
            for c in tmp_contours:
                tmp_m = np.zeros_like(mask)
                cv2.fillPoly(tmp_m, [c], 1)
                tmp_m = dilate(tmp_m, coef + 20)
                tmp_m = erode(tmp_m, 30)  # 27 worked fine, if you want to increase areas of farms
                                          # (and increase overlapping polygons), decrease erosion num
                tmp_mask_new += tmp_m
            polygons.append(c)
            filler += tmp_mask_new
        elif len(tmp_contours) == 0:
            continue
        else:
            # fill big and med holes
            separator = dilate(separator, 60)
            separator = erode(separator, 25)  # 20-30 should be fine
            filler += separator
    return filler

