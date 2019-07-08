import os
import numpy as np
import cv2
import urllib.request
import datetime
from PIL import Image
from tqdm import tqdm
from field_boundaries.utils.map_utils import LatLngTranslator, center_to_url
from pycocotools import mask


def polygons_to_union(image, polygons, return_mask=False):
    """ Create an union from polygons in order to proceed with crop function

        That method is critical for working with diamond-type areas
    """
    # reformat polygons to points to be like opencv
    points_as_opencv = [item for sublist in polygons for item in sublist]
    points_as_opencv = np.array(points_as_opencv).astype(np.int32)

    hull = cv2.convexHull(points_as_opencv)
    if return_mask:
        drawing = image.copy()
        color = (255, 255, 255)
        thickness = 1

        # draw lines for each point
        for i in range(len(hull)):
            start = tuple(hull[i][0])
            if i == (len(hull) - 1):
                i = -1
            end = tuple(hull[i + 1][0])
            cv2.line(drawing, start, end, color, thickness)
            return hull, drawing

    return hull


def region_to_pixel_v2(region, center, image, side=1280, pad=320, zoom=18, min_area=800,
                       split=False, with_diamonds=False):
    """ Translates all lat/long of the region to pixel distance using center as an anchor and crops the region

    v2: Improves crop via:
        a. padding an zeros_like_image_array with n=pad pixels each side, add the same value to all the points
        b. dropping/clipping all outliers
        c. processing with fill polygons algorithm
        d. cropping image to the original shape
        e. find contours

       Arguments:
           side (int): len of the side of area (h or w of rgb image)
           pad (int): pad each side of the image in order to get accurate crops
           zoom (int): google API zoom size, 15 for europe, 18 for pakistan
           min_area (int): area threshold to be used as segmentation
           split (bool): whether to split image on the 4 sub parts or not
           with_diamonds (bool): unstable method, to be tested further
       """
    # 1. Translate Lat Lng to pixels and add padding value to all the pixels
    translator = LatLngTranslator(center[0], center[1], zoom=zoom)
    pix_points = []
    areas = []
    for farm in region:
        f = []
        for point in farm:
            x, y = point[0], point[1]
            h, w = translator.latlng2local_pix(x, y)
            f.append([h + pad, w + pad])

        # 2. Drop outliers (even 1 point negative lies outside, consider as outlier)

        all_x_neg = all((p[0] >= 0) for p in f)
        all_x_out = all((p[0] < side + pad*2) for p in f)
        all_y_neg = all((p[1] >= 0) for p in f)
        all_y_out = all((p[1] < side + pad*2) for p in f)
        if all(z for z in [all_x_neg, all_x_out, all_y_neg, all_y_out]):
            # TODO rewrite vanilla clipping
            area = cv2.contourArea(np.array(f, dtype =np.int32))
            if area > min_area:
                pix_points.append(f)
                areas.append(area)

    # if the list is empty:
    if not pix_points:
        return [], []

    # remove duplicates
    _, indices = np.unique(areas, return_index=True)
    pix_points = list(np.array(pix_points)[indices])

    if split:
        contours, images = split_image(pix_points, image, pad, scale=2, new_side= 1024, min_area=min_area)

        return contours, images

    else:
        contours = []
        # 3. create and empty array like the original image, pad it, and fill it with polygons for each object
        for pt in pix_points:
            binary_map = np.zeros((side + pad * 2, side + pad * 2))
            cv2.fillPoly(binary_map, [np.array(pt, dtype=np.int32)], 1)

            # 4. crop segmentation map to original shape
            binary_map = binary_map[pad:side + pad, pad:side + pad]

            # 5. find polygons from the binary map in order to crop image according to boundary box
            _, contour, _ = cv2.findContours(binary_map.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contour:
                contours.append(contour)

        # 6.finding rectangle coordinates to crop an image and to clip h/w coordinates
        box_coord = [600, 600, 600, 600]  # [x0 x1 y0 y1]
        for contour in contours:
            x0, y0 = np.min(contour[0], axis=0)[0]
            x1, y1 = np.max(contour[0], axis=0)[0]
            box_coord[0] = min(x0, box_coord[0])
            box_coord[1] = max(x1, box_coord[1])
            box_coord[2] = min(y0, box_coord[2])
            box_coord[3] = max(y1, box_coord[3])

        # 7. Crop image
        image = image[box_coord[2]:box_coord[3] + 1, box_coord[0]:box_coord[1] + 1]

        # 8. correct contours according to bbox
        for contour in contours:
            contour[0][:, :, 0] -= box_coord[0]
            contour[0][:, :, 1] -= box_coord[2]
        contours = [contour[0].squeeze() for contour in contours]

    # 9. diamond shape processing # TODO find out stable method or delete part
    if with_diamonds:
        print("do not use it, not working properly")
        # hull = polygons_to_union(binary_map, contours)
        #
        # # convert back from opencv format
        # hull = [list(x[0]) for x in hull]
        #
        # # find maximal rectangle with the maximum area from the labeled region
        # try :
        #     ll, ur = get_maximal_rectangle(hull)
        # except:
        #     return contours, image, binary_map
        #
        # ll = [int(x) for x in ll]
        # ur = [int(x) for x in ur]
        #
        # binary_map = np.zeros_like(binary_map)
        # cv2.fillPoly(binary_map, contours, 1)
        #
        # # crop image and binary map
        # image = image[ll[1]:ur[1], ll[0]:ur[0]]
        # binary_map = binary_map[ll[1]:ur[1], ll[0]:ur[0]]
        #
        # # find contours
        # _, contours, _ = cv2.findContours(binary_map.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # check whether contours are containing more then 2 points
    contours = [x for x in contours if len(x) > 2]

    return [contours], [image]  # TODO probably remove


def split_image(polygons, image, pad, scale=2, new_side=1024, min_area=1000):
    """ Split Image and segmentation (polygons) to smaller equal fragments

    For example for image of shape 1280 x 1280 with scale 2 will be splitted to 4 512x512 images/segments

    Arguments:
        image (array): input image
        pad (int): pad each side of the image in order to get accurate crops (taken from the parent function)
        scale (int): difference between original image side and scaled one
        new_side (int): size of cropped image to be used for scaling
        min_area (int): min size of usable segmentation
    """
    side = image.shape[0]
    ad_crop = (side % new_side)//2

    # center crop image to the new_side
    image = image[ad_crop:-ad_crop, ad_crop:-ad_crop]

    # init polygons containers
    contours = [[] for _ in range(scale ** 2)]
    for pt in polygons:

        # 1. filter by area size
        binary_map = np.zeros((side + pad * 2, side + pad * 2))
        cv2.fillPoly(binary_map, [np.array([pt], dtype=np.int32)], 1)

        # erode mask a little bit
        kernel = np.ones((5, 5), np.uint8)
        binary_map = cv2.erode(binary_map, kernel, iterations=1)

        # 2. crop segmentation map from pad and ad_crop (to get new_side x new_side image)
        binary_map = binary_map[pad + ad_crop:side + pad - ad_crop, pad + ad_crop:side + pad - ad_crop]

        # 3. apply cropping window to label map
        step = new_side//scale
        for i in range(scale):
            for j in range(scale):
                sub_region = binary_map[i*step:(i+1)*step, j*step:(j+1)*step]
                _, contour, _ = cv2.findContours(sub_region.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # check for empty lists
                if not contour:
                    continue

                # check whether contours are containing more then 2 points
                contour = [x for x in contour if len(x) > 2]
                contours[(scale * i) + j].append(contour)

    # 4. crop image to sub parts
    images = [[] for _ in range(scale ** 2)]
    for i in range(scale):
        for j in range(scale):
            sub_region = image[i * step:(i + 1) * step, j * step:(j + 1) * step]
            images[(scale * i) + j] = (sub_region)

    # 5.finding rectangle coordinates to crop an image and to clip h/w coordinates
    for idx in range(len(contours)):
        box_coord = [-1, -1, -1, -1]
        sub = contours[idx]

        # if sub is empty workaround
        sub = [s for s in sub if s]

        subim = images[idx]
        init = 1  # gate to initiate box_coord from the values
        for contour in sub:
            x0, y0 = np.min(contour[0], axis=0).astype(int)[0]
            x1, y1 = np.max(contour[0], axis=0).astype(int)[0]
            if init:
                box_coord = [x0, x1, y0, y1]
                init = 0
            box_coord[0] = min(x0, int(box_coord[0]))
            box_coord[1] = max(x1, int(box_coord[1]))
            box_coord[2] = min(y0, int(box_coord[2]))
            box_coord[3] = max(y1, int(box_coord[3]))

        # 6. crop image
        images[idx] = subim[box_coord[2]:box_coord[3] + 1, box_coord[0]:box_coord[1] + 1]

        # 7. correct contours according to bbox
        for contour in sub:
            contour[0][:, :, 0] -= box_coord[0]
            contour[0][:, :, 1] -= box_coord[2]

        contours[idx] = [contour[0].squeeze() for contour in sub if cv2.contourArea(contour[0]) > min_area]

    return contours, images


def polygons2coco(polygons, image):
    """
        Converts polygons of farms on the image to annotation
    with bboxes, area, and encoded binary segmentation for each object

        Returns:
            ecncoded_masks, their  areas, their boundary boxes
    """

    # prepare polygons in COCO API format
    coco_poly = []
    for farm in polygons:
        topleft = np.min(farm, axis=0)
        botright = np.max(farm, axis=0)
        h, w = botright - topleft
        coco_poly.append([farm, h, w])

    # Convert polygon to encoded RLE mask, compute area and bbox
    bboxes = []
    segmentations = []
    # polys = []  tmp we are not going to use them
    areas = []
    for pts, h, w in coco_poly:
        zero_mask = np.zeros_like(image).astype(np.uint8)
        pts = np.array([pts], dtype=np.int32)
        cv2.fillPoly(zero_mask, pts, 1)
        fortran_binary_mask = np.asfortranarray(zero_mask)
        encoded_mask = mask.encode(fortran_binary_mask)  # 'size and 'counts
        encoded_mask['counts'] = encoded_mask['counts'].decode("utf-8")  # json-friendly
        area = mask.area(encoded_mask)  # 'area'
        if area < 800:
            continue
        bounding_box = mask.toBbox(encoded_mask)  # 'bbox'
        segmentations.append(encoded_mask)
        bboxes.append(bounding_box)
        areas.append(int(area))

    return segmentations, np.array(bboxes).astype(np.uint8).tolist(), areas


def create_annotation_info(ann_id, image_id, segmentation, bbox, gt_class, area, is_crowd=0):
    """ Information for the field 'annotations' of data dictionary"""

    annotation_info = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": gt_class,
            "bbox": bbox,
            "segmentation": segmentation,
            "iscrowd": is_crowd,                    # is crowd meaning that there are many subobjects in the area
            "area": area                            # is_crowd is irrelevant for the moment, set to 0
    }

    return annotation_info


def create_image_info(image_name, image_id, h, w):
    """ Information for the field 'images' of data dictionary"""
    image_info = {
        "file_name": image_name,
        "height": h,
        "width": w,
        "id": image_id
    }

    return image_info


def create_grid(center, output_path, zoom=18, grid_side_len=3):
    """ Function that takes a huge area around some center and returns RGB image of size 3840 x 3840

        Arguments:
            center (lat, long; float): center point of the future area
            output_path (str): path to directory for downloaded map images
            zoom (int): resolution of the map fragments of the future image
            grid_side_len (int): how many images along one side to use
    """
    # tmp
    center = center.split(',')
    center = [float(center[0]), float(center[1])]

    # initiate translator of lat/long to pix and vice versa
    translator = LatLngTranslator(center[0], center[1], zoom=zoom)
    c_lat, c_long = 640, 640

    # init grid
    h = translator.h
    grid = np.zeros((h * grid_side_len, h * grid_side_len, 3))

    # now we want to make 1280 pixels steps in 8 directions in order to get locations
    # within 1280 pixels distance from then center
    s = grid_side_len//2
    steps = [(-1280 * s) + 1280 * x for x in range(grid_side_len)]

    # start from top left corner
    for i in tqdm(range(grid_side_len)):
        for j in range(grid_side_len):
            x = c_lat + steps[i]
            y = c_long + steps[j]
            center = translator.local_pix2latlng(x, y)

            # get url and download map fragment
            url_image = center_to_url(center)
            key = i*grid_side_len + j
            urllib.request.urlretrieve(url_image, "{}/{}.png".format(output_path, key))
            image = cv2.imread("{}/{}.png".format(output_path, key))

            # bgr to rgb
            image = image[:, :, ::-1]

            # insert map fragment at the right place
            coef = 1280 * (s-1)
            x0 = x + 640 + coef
            y0 = y + 640 + coef
            x1 = x + 1920 + coef
            y1 = y + 1920 + coef
            grid[y0:y1, x0:x1, :] = image

    # save grid
    date = datetime.datetime.now().strftime('%m-%d_%H:%M')
    Image.fromarray(grid.astype(np.uint8)).save("{}/{}.jpg".format(output_path, date))
    print("file {}.jpg is ready to be used".format(date))


def create_grid_prod(points, output_path, name, pad=21, max_grid_side=4, zoom=18, image_side=1280):
    """
        Function that takes rectangle determined by 2 points (x0,y0) and (x1,y1) and creates stack of
            Google Maps Static Images of some specific zoom resolution which we call "grids"

        Arguments:
            points (list): list of next format [x0,y0,x1,y1] where x0y0 is top left point and x1y1 is bot right point
            output_path (str): path to directory for downloaded map images
            name (int): name id of the output grid file
            pad (int): pad rectangle in order to crop it further, do not change it or you will have to change crop
                       size in "ricult_boundary_engine" script
            max_grid_side (int): max number of images along one side of the grid, 4 is default, do not change it
                                 or you will have to change padding function in "ricult_boundary_engine" script
            zoom (int): resolution of the map fragments of the future image
            image_side (int): grid is produced out of square images of some side, default is 1280, do not change it
                              or you will have to add non-default variables to the "translator" function
        Returns:
            list of names of grids produced from initial 2 point rectangle,
            center of the rectangle to be used to convert polygons from pixel space to lat/long,
            h_cor, w_cor: constants in order to scale pix values of grid to get lat/long
                          using the same center value for all the grids
    """
    # find center of the rectangle
    x0, y0, x1, y1 = points
    center = [np.mean((x0, x1)), np.mean((y0, y1))]
    max_pix = max_grid_side * image_side

    # initiate translator of lat/long to pix and vice versa
    translator = LatLngTranslator(center[0], center[1], zoom=zoom)

    # the easiest way to get static map image is to via center point of the original rectangle
    c_lat, c_long = image_side//2, image_side//2

    # find pix coordinates as distance from the center of the rectangle
    x0, y0 = translator.latlng2local_pix(x0, y0)
    x1, y1 = translator.latlng2local_pix(x1, y1)

    # pad coordinates in order to crop them
    x0, y0 = x0 - pad - c_lat, y0 - pad - c_long
    x1, y1 = x1 + pad - c_lat, y1 + pad - c_long

    # find min (negative) val and change it's sign to plus in order to find our rectangle area shape
    h = x1 - x0
    w = y1 - y0

    # split rectangle to grids of max max_pix x max_pix size if it is possible
    h_grid = 1
    w_grid = 1
    if h/max_pix > 1:
        if h % max_pix == 0:
            h_grid = h // max_pix
            h_grid_out = 0
        else:
            h_grid = h//max_pix + 1
            h_grid_out = h % max_pix
    if w/max_pix > 1:
        if w % max_pix == 0:
            w_grid = w // max_pix
            w_grid_out = 0
        else:
            w_grid = w//max_pix + 1
            w_grid_out = w % max_pix

    # initiate grid names and constants container
    grids = []

    # iterate over grid spaces of original rectangle
    for a in range(h_grid):
        for b in range(w_grid):

            # init height and width
            height = h if h_grid == 1 else max_pix
            width = w if w_grid == 1 else max_pix

            # correct empty grid initial coordinate space according to index of h_grid / w_grid
            h_cor = x0 + max_pix * a
            w_cor = y0 + max_pix * b

            # init grid
            grid = np.zeros((height, width, 3))

            # find number of rows, columns of images to be downloaded
            h_fragments = height // image_side
            w_fragments = width // image_side

            # start from top left corner, download images and build a grid
            for i in range(h_fragments):
                for j in range(w_fragments):
                    x = h_cor + image_side*i + c_lat
                    y = w_cor + image_side*j + c_long
                    center = translator.local_pix2latlng(x, y)

                    # get url and download map fragment
                    url_image = center_to_url(center)
                    key = "{}_{}_{}_{}_{}".format(name, a, b, i, j)
                    path = "{}/{}.png".format(output_path, key)
                    urllib.request.urlretrieve(url_image, path)
                    image = cv2.imread(path)

                    # bgr to rgb
                    image = image[:, :, ::-1]

                    # insert map fragment at the right place
                    h0 = i * image_side
                    w0 = j * image_side
                    h1 = (i+1) * image_side
                    w1 = (j+1) * image_side
                    grid[w0:w1, h0:h1, :] = image

                    # delete the image been already used
                    os.remove(path)

            # save grid
            grid_name = "grid_{}_{}".format(name, w_grid * a + b)
            grid_path = "{}/{}.png".format(output_path, grid_name)
            Image.fromarray(grid.astype(np.uint8)).save(grid_path)
            # print("file {} is ready to be used".format(grid_path))
            grids.append([grid_path, h_cor, w_cor])

    return grids, center, h_grid_out, w_grid_out


if __name__ == "__main__":
    # Parse program arguments
    import argparse

    parser = argparse.ArgumentParser("downloading images using SQL script")
    parser.add_argument('--points',
                        help='x0,y0,x1,y0')
    parser.add_argument('--output_path',
                        help='path to the directory to save images in')
    parser.add_argument('--name', help='name of the output file')
    args = parser.parse_args()

    points = args.points.split(',')
    points = [float(p) for p in points]
    create_grid_prod(points=points, output_path=args.output_path, name=args.name)
