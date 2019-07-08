import csv
import re
import numpy as np
import math
from pyproj import Proj, transform


class LatLngTranslator:

    """ Class that takes care of translation of latitude and longitude to pixel space and vice versa"""

    def __init__(self, center_lat, center_lng, tile_size=256, scale=2.0, zoom=18, h=1280, w=1280):
        self.tile_size = tile_size
        self.scale = scale
        self.zoom = zoom
        self.multiplier = self.scale * (2 ** self.zoom)
        self.center_lat = center_lat
        self.center_lng = center_lng
        self.origin = self.get_origin(center_lat, center_lng)
        self.h = h
        self.w = w

    def latlng2w(self, latitude, longitude):

        """ Translates Latitude and Longitude to World Coordinates"""

        # // The mapping between latitude, longitude and pixels is defined by the web
        # // mercator projection.

        siny = math.sin(latitude * math.pi / 180)

        #   // Truncating to 0.9999 effectively limits latitude to 89.189. This is
        #   // about a third of a tile past the edge of the world tile.
        siny = min(max(siny, -0.9999), 0.9999)

        x = self.tile_size * (0.5 + longitude / 360)
        y = self.tile_size * (0.5 - math.log((1 + siny) / (1 - siny)) / (4 * math.pi))

        return x, y

    def w2pix(self, x, y):

        """ Translates World Coordinates to global pixel values"""

        height = math.floor(x * self.multiplier)
        width = math.floor(y * self.multiplier)

        return height, width

    def pix2w(self, height, width):

        """ Translates global pixel values to World Coordinates"""

        x = height / self.multiplier
        y = width / self.multiplier

        return x, y

    def w2tile(self, x, y):

        """ Translates  World Coordinates to tile coordinates"""

        x = math.floor(x * self.multiplier / self.tile_size)
        y = math.floor(y * self.multiplier / self.tile_size)

        return x, y

    def tile2w(self, x, y):

        """ Translates  tile coordinates to World Coordinates  """

        x = x * self.tile_size / self.multiplier
        y = y * self.tile_size / self.multiplier

        return x, y

    def get_origin(self, center_lat, center_lng):

        """ Calculates origin from global height and width of the center point"""
        world_x, world_y = self.latlng2w(center_lat, center_lng)
        center_pix_x, center_pix_y = self.w2pix(world_x, world_y)

        x0 = center_pix_x - 640  # 639?
        y0 = center_pix_y - 640   # 639?

        return x0, y0

    def latlng2local_pix(self, latitude, longitude):

        """ Translates latitude and longitude to local pixel distance according to the origin"""

        world_x, world_y = self.latlng2w(latitude, longitude)
        glob_pix_x, glob_pix_y = self.w2pix(world_x, world_y)
        loc_pix_x, loc_pix_y = glob_pix_x - self.origin[0], glob_pix_y - self.origin[1]

        return loc_pix_x, loc_pix_y

    def local_pix2latlng(self, x, y):

        """ Translates local_pixel values to latitude and longitude according to lat/long of center of the image"""

        parallel_multiplier = math.cos(self.center_lat * math.pi / 180)
        degrees_x = 360 / (2 * math.pow(2, self.zoom + 8))
        degrees_y = parallel_multiplier * 360 / (2 * math.pow(2, self.zoom + 8))
        latitude = self.center_lat - degrees_y * (y - self.h / 2)
        longitude = self.center_lng + degrees_x * (x - self.w / 2)

        return latitude, longitude


def all_csv_to_latlng(csv_file):

    """ Retrieve Latitude and Longitude and center of the region from CSV file from contatining all valid users"""

    csvfile = open(csv_file, newline='')
    data = csv.reader(csvfile)

    # skip the headers
    next(data, None)

    # init tmp dic to get floats out of str
    regions = {}

    # dic keys
    for row in data:
        coords = re.findall(r'\[{1}[^\[]+\]{1}', row[12][1:-1])
        float_coords = []
        for i in range(len(coords)):
            points = re.findall(r'\d+\.\d+', coords[i])  # find all floats
            points = [float(z) for z in points]
            float_coords.append(points)

        region_id = row[8]
        if region_id not in regions.keys():
            regions[region_id] = [float_coords]
        regions[region_id].append(float_coords)

    centers = {}
    for key, val in regions.items():
        means = []
        for farm in val:
            means.append(np.mean(farm, axis=0, dtype=np.float64))
        centers[key] = np.mean(means, axis=0, dtype=np.float64)

    return regions, centers


def coordinates_to_url(region , center, zoom=18, scale=2, size=(640, 640), weight=1, use_center=True,
                       key='AIzaSyBiZd8etUKtX2EW8_lVaQNnZAuPNiLrewg', color='0xffffffff', fillcolor=False):

    """ Make valid url based on GOOGLE MAPS API in order to retrieve an image of the region with farm borders

    Arguments:
        zoom (int): zoom value for google maps API
        scale (1 or 2): whether to scale image in order to get x2 pixels value for the area or not
        size (tuple of ints): relative size of the area (up to 640)
        weight (int): thickness of the line (0 for test data, 1 for train)
        use_center (bool): whether we want to center image around some coordinate or not
        key (str): key in order to use Google Maps API
        color (str): color of the line, by default white color without transparency
        fillcolor (bool): True, if we want polygons to be filled with color, False for borders only

    Returns:
        url of the image of the region
    """
    fill = ''
    if fillcolor:
        fill = '|fillcolor:{}'.format(color)
    prefix = 'https://maps.googleapis.com/maps/api/staticmap?'
    if use_center:
        prefix += 'center={},{}'.format(center[0], center[1])
    mid = ''
    for farms in region:
        str_ = '&path=color:{}|weight:{}{}'.format(color, weight, fill)
        for point in farms:
            str_ += "|{:.6f},{:.6f}".format(point[0], point[1])

        # in order to get polygon closed, you have to repeat first point
        str_ += "|{:.6f},{:.6f}".format(farms[0][0], farms[0][1])

        # update middle part of url
        mid += str_
    suffix = '&zoom={}&scale={}&size={}x{}&maptype=satellite&key={}'.format(zoom, scale, size[0], size[1], key)

    return prefix + mid + suffix


def polygon_to_url(farm, zoom=18, scale=2, size=(640, 640), weight=1,
                   key='AIzaSyDgafXr2goOTxrU0_JYLVlkyI3QofQmWMo', color='0xffffffff'):
    """ Make valid url based on GOOGLE MAPS API in order to retrieve an image of the region with farm borders

    Arguments:
        zoom (int): zoom value for google maps API
        scale (1 or 2): whether to scale image in order to get x2 pixels value for the area or not
        size (tuple of ints): relative size of the area (up to 640)
        weight (int): thickness of the line (0 for test data, 1 for train)
        key (str): key in order to use Google Maps API
        color (str): color of the line, by default white color without transparency

    Returns:
        url of the image of the region
    """

    prefix = 'https://maps.googleapis.com/maps/api/staticmap?'
    mid = ''
    str_ = '&path=color:{}|weight:{}'.format(color, weight)
    for point in farm:
        str_ += "|{:.6f},{:.6f}".format(point[0], point[1])

    # in order to get polygon closed, you have to repeat first point
    str_ += "|{:.6f},{:.6f}".format(farm[0][0], farm[0][1])

    # update middle part of url
    mid += str_
    suffix = '&zoom={}&scale={}&size={}x{}&maptype=satellite&key={}'.format(zoom, scale, size[0], size[1], key)

    return prefix + mid + suffix


def center_to_url(center, zoom=18, scale=2, size=(640, 640), key='AIzaSyDgafXr2goOTxrU0_JYLVlkyI3QofQmWMo'):
    """ Make valid url based on GOOGLE MAPS API from a central coordinate in order to retrieve image around the point

        Check API of the function above
    """
    prefix = 'https://maps.googleapis.com/maps/api/staticmap?center={},{}'.format(center[0], center[1])
    suffix = '&zoom={}&scale={}&size={}x{}&maptype=satellite&key={}'.format(zoom, scale, size[0], size[1], key)

    return prefix + suffix


def image_to_latlng(polygons, center, zoom=18):

    """ Translates all pixel point of the image around some center (lat, long) to lat/long points"""

    translator = LatLngTranslator(center[0], center[1], zoom)
    latlng_points = []
    for farm in polygons:
        f = []
        for point in farm:
            x, y = point[0], point[1]
            lat, long = translator.local_pix2latlng(x, y)
            f.append([lat, long])
        latlng_points.append(f)

    return latlng_points


def region_to_pixel(region, center, side=1280, zoom=18):

    """ Translates all lat/long of the region to pixel distance using center as an anchor

        Arguments:
            side (float): len of the side of area (h or w of rgb image)
    """

    translator = LatLngTranslator(center[0], center[1], zoom)
    pix_points = []
    for farm in region:
        f = []
        for point in farm:
            x, y = point[0], point[1]
            h, w = translator.latlng2local_pix(x, y)
            f.append([h, w])

        # if all x or y coordinates lie out of [0:1280] -> throw farm
        all_x_neg = all((p[0] < 0) for p in f)
        all_x_out = all((p[0] > side) for p in f)
        all_y_neg = all((p[1] < 0) for p in f)
        all_y_out = all((p[1] > side) for p in f)
        if all(not z for z in [all_x_neg, all_x_out, all_y_neg, all_y_out]):
            # TODO rewrite vanilla clipping
            pix_points.append(np.clip(f, 0, side - 1))
        else:
            continue

    return pix_points


def transform_proj(x, y, proj_in='epsg:25832', proj_out='epsg:4326', lpis=True):
    """ transforms x y to lat/long (default values used for Denmark LPIS dataset"""
    inProj = Proj(init=proj_in)
    outProj = Proj(init=proj_out)
    lat, lng = transform(inProj, outProj, x, y)
    if lpis:
        # swap lat and lng in order to be similiar with google maps
        lng, lat = (lat, lng)

    return lat, lng
