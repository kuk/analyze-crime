

import os
import re
import sys
import json
from math import sqrt
from datetime import datetime
from collections import namedtuple

from random import seed as random_seed
from random import sample

from tqdm import tqdm_notebook as log_progress

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import seaborn as sns

from bokeh import plotting as bk
from bokeh.tile_providers import WMTSTileSource as TileSource

import xarray as xr

import datashader.transfer_functions as tf
from datashader.colors import inferno

import pyproj


DATA_DIR = 'data'
SUMMARY = os.path.join(DATA_DIR, 'summary.csv')
NARKOTA = os.path.join(DATA_DIR, 'narkota.csv')

WIDTH = 700
LIGHT_CARTO = TileSource(
    url='http://tiles.basemaps.cartocdn.com/light_all/{z}/{x}/{y}@2x.png'
)
DARK_CARTO = TileSource(
    url='http://tiles.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}@2x.png'
)
YANDEX = TileSource(
    url='https://vec02.maps.yandex.net/tiles?l=map&x={x}&y={y}&z={z}&scale=2',
    y_origin_offset=20002108.34
)


SummaryRecord = namedtuple(
    'SummaryRecord',
    ['date', 'group', 'type', 'victims', 'dead', 'cell']
)
NarkotaRecord = namedtuple(
    'NarkotaRecord',
    ['date', 'sbit_try', 'hran', 'storage', 'other', 'cell']
)
Point = namedtuple(
    'Point',
    ['latitude', 'longitude']
)
Cell = namedtuple(
    'Cell',
    ['corner', 'points']
)
BoundingBox = namedtuple(
    'BoundingBox',
    ['lower_left', 'upper_right']
)


MOSCOW_BOX = BoundingBox(
    Point(55.555959, 37.252433),
    Point(55.929357, 37.952812)
)


def parse_cell(cell):
    if not cell:
        return
    cell = json.loads(cell)
    points = tuple(
        Point(latitude, longitude)
        for latitude, longitude in cell
    )
    corner = points[0]
    return Cell(corner, points)


def parse_date(date):
    return datetime.strptime(date, '%Y-%m-%d')


def load_summary():
    table = pd.read_csv(SUMMARY, sep=';')
    table = table.where(table.notnull(), None)
    table.columns = ['index', 'date', 'group', 'type', 'victims', 'dead', 'cell']

    for record in table.itertuples():
        date = parse_date(record.date)
        group = re.sub('\s+', ' ', record.group)
        cell = parse_cell(record.cell)
        yield SummaryRecord(
            date, group, record.type,
            record.victims, record.dead,
            cell
        )


def is_inside(point, box):
    return (
        box.lower_left.latitude < point.latitude < box.upper_right.latitude
        and box.lower_left.longitude < point.longitude < box.upper_right.longitude
    )


def show_cell_corners(cells, box):
    xs = [_.corner.longitude for _ in cells]
    ys = [_.corner.latitude for _ in cells]
    fig, ax = plt.subplots()
    fig.set_size_inches([5, 5])
    ax.set_xlim([box.lower_left.longitude, box.upper_right.longitude])
    ax.set_ylim([box.lower_left.latitude, box.upper_right.latitude])
    ax.scatter(xs, ys, lw=0, s=4)


def get_cell_xys(cells):
    latitudes = set()
    longitudes = set()
    for cell in cells:
        latitude, longitude = cell.corner
        latitudes.add(latitude)
        longitudes.add(longitude)
    latitude_ys = {}
    for y, latitude in enumerate(sorted(latitudes, reverse=True)):
        latitude_ys[latitude] = y
    longitude_xs = {}
    for x, longitude in enumerate(sorted(longitudes)):
        longitude_xs[longitude] = x
    for cell in cells:
        latitude, longitude = cell.corner
        x = longitude_xs[longitude]
        y = latitude_ys[latitude]
        yield cell, x, y


def get_xys_width_height(cell_xys):
    xys = cell_xys.values()
    width = max(x for x, y in xys) + 1
    height = max(y for x,y in xys) + 1
    return width, height


def get_cell_series_heatmap(series, cell_xys, log=True):
    width, height = get_xys_width_height(cell_xys)
    heatmap = np.zeros([height, width])
    for cell, value in series.iteritems():
        if cell in cell_xys:
            x, y = cell_xys[cell]
            heatmap[height - y - 1][x] = value
    if log:
        heatmap = np.log(heatmap + 1)
    return heatmap


def shorten(string, cap=50):
    if len(string) > cap:
        return string[:cap] + '...'
    return string


def plot_heatmap(heatmap, ax=None):
    if not ax:
        fig, ax = plt.subplots()
    ax.imshow(heatmap, cmap='inferno')
    ax.grid('off')
    ax.axis('off')


def show_group_maps(table, cell_xys):
    fig, axis = plt.subplots(2, 2)
    groups = [
        'Глава 21. Преступления против собственности',
        ('Глава 25. Преступления против здоровья населения '
         'и общественной нравственности'),
        'Глава 32. Преступления против порядка управления',
        'Глава 16. Преступления против жизни и здоровья'
    ]
    for ax, group in zip(axis.flatten(), groups):
        view = table[table.group == group]
        view = view.groupby('cell').size()
        heatmap = get_cell_series_heatmap(view, cell_xys)
        plot_heatmap(heatmap, ax=ax)
        ax.set_title(shorten(group))
    fig.set_size_inches([10, 10])
    fig.tight_layout(pad=0)


def load_narkota():
    table = pd.read_csv(NARKOTA, sep=';')
    table = table.where(table.notnull(), None)
    table.columns = ['index', 'date', 'sbit_try', 'storage', 'sbit', 'other', 'cell']
    for record in table.itertuples():
        date = parse_date(record.date)
        cell = parse_cell(record.cell)
        yield NarkotaRecord(
            date,
            record.sbit_try, record.storage,
            record.sbit, record.other,
            cell
        )


WORLD = pyproj.Proj(init='epsg:4326')
MERCATOR = pyproj.Proj(init='epsg:3857')


def convert_point(point, source=WORLD, target=MERCATOR):
    try:
        longitude, latitude = pyproj.transform(
            source, target,
            point.longitude, point.latitude
        )
    except RuntimeError:
        raise ValueError(point)
    return Point(latitude, longitude)


def convert_box(box):
    return BoundingBox(
        convert_point(box.lower_left),
        convert_point(box.upper_right)
    )


def get_figure(box, width=WIDTH):
    box = convert_box(box)
    y_min = box.lower_left.latitude
    y_max = box.upper_right.latitude
    x_min = box.lower_left.longitude
    x_max = box.upper_right.longitude
    x_range = (x_min, x_max)
    y_range = (y_min, y_max)
    height = int(width / (x_max - x_min) * (y_max - y_min))
    
    figure = bk.figure(
        tools='pan, wheel_zoom',
        active_scroll='wheel_zoom',
        plot_width=width, plot_height=height,
        x_range=x_range, y_range=y_range,
    )
    figure.axis.visible = False
    return figure


def get_map(box, tiles, width=WIDTH):
    figure = get_figure(box, width=width)
    figure.add_tile(tiles)
    return figure
