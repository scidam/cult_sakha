# -*- coding:utf-8 -*-
"""
Created Date: Tuesday March 3rd 2020
Author: Dmitry Kislov
E-mail: kislov@easydan.com
-----
Last Modified: Sunday, August 23rd 2020, 2:44:26 pm
Modified By: Dmitry Kislov
-----
Copyright (c) 2020
"""

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
      print("OK")
  except RuntimeError as e:
    print(e)


from model import *
import os
import numpy as np
from skimage.io import imsave
from skimage.util import montage, view_as_windows
from scipy import ndimage
from PIL import Image
import gdal
import osr
import gc


# os.environ['PROJ_LIB'] = '/home/dmitry/bin/gdal/share/proj'
CHUNK_SIZE = 256 * 10


def read_by_chunk(file, chunksize=CHUNK_SIZE):
    print(f"Opening file {file} for reading... ")
    df = gdal.Open(file)
    rb = df.GetRasterBand(1)
    print(f"Band sizes: {rb.XSize}, {rb.YSize}")
    nchunks_x = rb.XSize // chunksize
    nchunks_y = rb.YSize // chunksize
    print(f"Grid sizes: x-chunks={nchunks_x}, y-chunks={nchunks_y}")
    for xcol in range(nchunks_x):
        for ycol in range(nchunks_y):
            dataframe = []
            for band_num in range(1, df.RasterCount + 1):
                rb = df.GetRasterBand(band_num)
                dataframe.append(rb.ReadAsArray(xcol * chunksize, ycol * chunksize, chunksize, chunksize))
            yield np.dstack(dataframe), xcol * chunksize, ycol * chunksize
        
        if rb.YSize % chunksize != 0:
            dataframe = []
            for band_num in range(1, df.RasterCount + 1):
                rb = df.GetRasterBand(band_num)
                dataframe.append(rb.ReadAsArray(xcol * chunksize, (ycol + 1) * chunksize, chunksize, rb.YSize % chunksize))
            yield np.dstack(dataframe), xcol * chunksize, (ycol + 1) * chunksize

    if rb.XSize % chunksize != 0:
        dataframe = []
        for band_num in range(1, df.RasterCount + 1):
            rb = df.GetRasterBand(band_num)
            dataframe.append(rb.ReadAsArray((xcol + 1) * chunksize, ycol * chunksize, rb.XSize % chunksize, chunksize))
        yield np.dstack(dataframe), (xcol + 1) * chunksize, ycol * chunksize


def GetGeoInfo(FileName):
    from gdalconst import GA_ReadOnly
    SourceDS = gdal.Open(FileName, GA_ReadOnly)
    NDV = SourceDS.GetRasterBand(1).GetNoDataValue()
    xsize = SourceDS.RasterXSize
    ysize = SourceDS.RasterYSize
    GeoT = SourceDS.GetGeoTransform()
    Projection = osr.SpatialReference()
    Projection.ImportFromWkt(SourceDS.GetProjectionRef())
    DataType = SourceDS.GetRasterBand(1).DataType
    DataType = gdal.GetDataTypeName(DataType)
    return NDV, xsize, ysize, GeoT, Projection, DataType



def write_by_chunk(file, xsize, ysize,  chunksize=CHUNK_SIZE, likefile=""):
    if not likefile:
        print("You need to set name of a file that will be used to extract information about the projection")
        return
    NDV, xsize, ysize, GeoT, Projection, DataType = GetGeoInfo(likefile)
    
    driver = gdal.GetDriverByName('gtiff')
    ds = driver.Create(f'{file}', xsize, ysize, 1, gdal.GDT_Float32)
    ds.SetGeoTransform(GeoT)
    ds.SetProjection(Projection.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(-9999.0)
    print(f"Preapared to save to {file}.")
    while True:
        array, xpos, ypos = yield
        print(f"Got the data: {array.shape}, {xpos}, {ypos}")
        if xpos < 0 or ypos < 0:
            band.FlushCache()
            break
        array[np.isnan(array)] = -9999.0
        band.WriteArray(array, xpos, ypos)
        band.FlushCache()
        print("The data were written.")
    ds = None

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    
    filenames = ['Dolinsky.tif',]
    model = unet(pretrained_weights='bugs_batchnorm=False_start_ch=18_depth=5_lr=1e-05.hdf5', lr=1e-5, start_ch=18, batchnorm=False, depth=5)

    window_shape = (256, 256, 3)

    # =============================================================================
    for fname in filenames:
        _, xsize, ysize, *args = GetGeoInfo(fname)
        writer = write_by_chunk(f'out_{fname}', CHUNK_SIZE, CHUNK_SIZE, likefile=fname)
        writer.send(None)
        print(f"Writer prepared for writing: {fname}...")
        for im, xpos, ypos in read_by_chunk(fname):
            if im.shape[0] < 256 or im.shape[1] < 256:
                continue
            print(f"Read chunks from {fname}, shape={im.shape}, xpos={xpos},ypos={ypos}")
            im = (im / 255).astype(np.float32)
            cimg = view_as_windows(im, window_shape, window_shape[0])
            orig_shape = cimg.shape
            predictions = model.predict(np.squeeze(cimg.reshape(-1, *cimg.shape[2:]), axis=1))#[...,-1]
            orig = montage(predictions, grid_shape=orig_shape[:2], multichannel=True)
            array_to_save = orig[..., -1]
            print(f"Array of probabilities prepared: {array_to_save.shape}")
            writer.send((array_to_save, xpos, ypos))
            gc.collect()
        try:
            writer.send((array_to_save, -1, -1))
        except StopIteration:
            print(f"Job done: {fname}.")





# Converting images to tiles using imagemagick.
# convert 1.png -crop 2560x2560 tile%03d.png
# montage -compress lzw -tile 8x15 -geometry +0+0 *.tiff output.tiff


