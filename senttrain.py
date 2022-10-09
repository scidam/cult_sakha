# train_sentinel data

import os
import re
import numpy as np
import pandas as pd
from functools import lru_cache
from osgeo import gdal
from osgeo import osr
from shapely.ops import cascaded_union, unary_union
import geopandas as gpd
from pprint import pprint
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, plot_confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.manifold import MDS
from sklearn.svm import SVC

from skimage.util import invert
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage import io


from skimage.io import imsave
from skimage.util import montage, view_as_windows
from scipy import ndimage
from PIL import Image
import gdal
import osr
import gc
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pyproj import Proj, transform
import seaborn as sns
import sys

from collections import defaultdict, Counter

CHUNK_SIZE = 256 * 4

#os.environ['PROJ_LIB'] = '/home/dmitry/bin/gdal/share/proj'


def convert_coords(lat, lon, inprj='epsg:4326', outprj='epsg:32654'):
    in_project = Proj(inprj)
    out_project = Proj(outprj)
    return transform(in_project, out_project, lat, lon)


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
        print("You need to set name of a file that will be used to extract"
              "information about the projection")
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


def traverse_folder(folder='./data/R10m/'):
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in files:
            fname = os.path.join(root, name)
            if fname.lower().endswith('jp2'):
                yield fname

# ------- Settings data ---------------------
LARGE_VALUE = 500



# --------- used for data loading -----------

def get_data_by_coordinate_np(lats, lons, array, xmin, xres, ymax, yres):
    lat_inds = ((lats - ymax) / yres).astype(np.int64)
    lon_inds = ((lons - xmin) / xres).astype(np.int64)
    lat_ind_max, lon_ind_max = array.shape
    lat_ind_max -= 1
    lon_ind_max -= 1
    mask_lat = (lat_inds >= 0) * (lat_inds <= lat_ind_max)
    mask_lon = (lon_inds >= 0) * (lon_inds <= lon_ind_max)
    full_mask = mask_lat * mask_lon
    _res = np.full_like(lats, np.nan)
    _res[full_mask] = array[lat_inds[full_mask], lon_inds[full_mask]]
    return _res


def get_arrays(level, patterns=None):
    result = list()
    for f in patterns[level]:
        data = gdal.Open(f)
        geoinfo = data.GetGeoTransform()
        xmin = geoinfo[0]
        xres = geoinfo[1]
        ymax = geoinfo[3]
        yrot = geoinfo[4]
        xrot = geoinfo[2]
        yres = geoinfo[-1]
        if not np.isclose(xrot, 0) or not np.isclose(yrot, 0):
            raise BaseException("xrot and yrot should be 0")
        array = data.ReadAsArray()
        del data
        if "20191018" in f:
            array = np.full_like(array, np.nan)
        result.append((array.astype(np.float64), xmin, xres, ymax, yres))

    return result


def get_data(lats, lons, levels, patterns=None):
    result = []
    for l in levels:
        intermediate = []
        for array, xmin, xres, ymax, yres in get_arrays(l, patterns):
            intermediate.append(get_data_by_coordinate_np(np.array(lats, dtype=np.float64),
                                np.array(lons, dtype=np.float64),
                                np.array(array, dtype=np.float64),
                                xmin, xres, ymax, yres)[:, np.newaxis])
        result.append(np.nanmean(np.hstack(intermediate), axis=1))
    return result


# -----------------------------------------------------------
# def rescale_image(img, scale=1.0):
#     if abs(scale - 1.0) < 1.e-6:
#         result = img
#     else:
#         imh, imw = img.shape[0], img.shape[1]
#         simh, simw = int(imh * scale), int(imw * scale)
#         result = resize(resize(img, (simh, simw), anti_aliasing=True), (imh, imw), anti_aliasing=True)

#     if (result > 10.0).any():
#         return (result / 255).astype(np.float32)
#     else:
#         return result


def make_training_points_csv(filename='pinus_larix_3.csv', step=30):
    LATS = []
    LONS = []
    Y = []
    info = dict()
    df = pd.read_csv(filename)
    cnt = 1
    for ind, item in df.iterrows():
        lon1 = item.lon_min
        lon2 = item.lon_max
        lat1 = item.lat_min
        lat2 = item.lat_max
        lon1, lat1 = convert_coords(lat1, lon1)
        lon2, lat2 = convert_coords(lat2, lon2)

        x, y = np.meshgrid(
                np.arange(lat1, lat2, step),
                np.arange(lon1, lon2, step)
            )
        if len(x.ravel()) > 1 and len(y.ravel()) > 1:
            LATS.append(x.ravel())
            LONS.append(y.ravel())
            Y.extend([item.species] * len(x.ravel()))
            info[cnt] = [lat1, lon1]
        else:
            print(f"error: {lat1}, {lat2}; {lon1}, {lon2}")
            print(f"{item.lon_min}")
        cnt += 1
    LATS, LONS = np.hstack(LATS), np.hstack(LONS)
    return LATS, LONS, Y, info


# def load_predict_data():
#     XP = []
#     for key in features:
#         data = gdal.Open(PREDICT_PATTERNS20m[key][0])
#         XP.append(data.ReadAsArray().ravel())
#         del data
#     return np.vstack(XP).T


def prepare_data_patterns(path):

    DATA_PATTERNS10m = defaultdict(list)

    for item in traverse_folder(folder=f'./{path}/R10m/'):
        key = item.split('/')[-1].split('.')[0].split('_')[-2]
        DATA_PATTERNS10m[key].append(item)

    DATA_PATTERNS20m = defaultdict(list)
    for item in traverse_folder(folder=f'./{path}/R20m/'):
        key = item.split('/')[-1].split('.')[0].split('_')[-2]
        DATA_PATTERNS20m[key].append(item)
    return DATA_PATTERNS10m, DATA_PATTERNS20m


def prepare_dataset(
    lats,
    lons,
    high_res_features,
    low_res_features, paths=['data'],
    filter_by_scl=[2, 3, 8, 9]
):
    X = []
    for path in paths:
        pat_high, pat_low = prepare_data_patterns(path)
        data_hr = get_data(lats, lons, high_res_features, patterns=pat_high)
        data_lr = get_data(lats, lons, low_res_features, patterns=pat_low)
        x_hr = np.vstack(data_hr).T
        x_lr = np.vstack(data_lr).T
        X1 = np.hstack([x_hr, x_lr])
        if filter_by_scl:
            filter_array = np.array(filter_by_scl)
            X1[np.in1d(X1[:, -1], filter_array), :] = np.nan
        X1 = X1[:, :-1]
        ndvi = (X1[:, 3] - X1[:, 2]) / (X1[:, 3] + X1[:, 2])
        ndwi = (X1[:, 4] - X1[:, 5]) / (X1[:, 4] + X1[:, 5])
        bsi = ((X1[:, 5] + X1[:, 2]) - (X1[:, 4] + X1[:, 0])) / ((X1[:, 5] + X1[:, 2]) + (X1[:, 4] + X1[:, 0]))
        X1 = np.hstack([X1, ndvi[:, np.newaxis], ndwi[:, np.newaxis], bsi[:, np.newaxis]])
        X.append(X1.copy())

    X = np.hstack(X).copy()
    return X


def plot_error_matrix(clf, X, y, labenc=None, feature_names=[]):
    from random import random
    if labenc:
        y_ = labenc.inverse_transform(y)
    else:
        y_ = y
    X_train, X_test, y_train, y_test = train_test_split(X, y_)
    clf.fit(X_train, y_train)
    plot_confusion_matrix(clf, X_test, y_test, xticks_rotation='vertical')
    plt.gcf().savefig(f"{str(random())}.png", dpi=600)
    print('='*80)
    print(f"Selected features {np.array(feature_names)[clf.support_]}")
    print(f"Ranking: {clf.ranking_}")
    print(f"Grid scores{clf.grid_scores_}")
    print('='*80)



def save_as_tiff(x, xsize, ysize, GeoT, fname='output', like="./data/R20m/T54TXT_20200601T012701_B02_20m.jp2"):
    NDV, _, _, _, Projection, DataType = GetGeoInfo(like)
    driver = gdal.GetDriverByName('gtiff')
    ds = driver.Create(f'{fname}.tif', xsize, ysize, 1, gdal.GDT_Byte)
    ds.SetGeoTransform(GeoT)
    ds.SetProjection(Projection.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(254)
    band.WriteArray(np.flipud(x.reshape(xsize, ysize).T))
    band.FlushCache()

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # print("Do testing ... ")
    # lats = (4861736, 4853744)
    # lons = (387195, 381430)

    # data = get_data(np.array(lats), np.array(lons), ['B02',], normalize=False)
    # print(data)


    CLASSIFIERS = [
        # ('tree', DecisionTreeClassifier(random_state=10)),
        # # ('NB', GaussianNB()),
        # ('MaxEnt', LogisticRegression()),
        ('RF_100', RandomForestClassifier(n_estimators=200, random_state=10)),
        # ('ada', AdaBoostClassifier(DecisionTreeClassifier(max_depth=7),
                                #    n_estimators=200, random_state=10)),
        # ('SVM', SVC(kernel='linear')),
        # ('LDA', LinearDiscriminantAnalysis()),
        # ('QDA', QuadraticDiscriminantAnalysis())
    ]


    # ---------------------- Training and validating model ------------------

    #features = ['B08', ]#, 'B08', 'B8A', 'B11', 'B12', 'B05', 'B06']
    # features = ['B8A', 'B11', 'B12'] # 'B8A', 'B11', 'B12', 'B05', 'B06']
    # high_res_features = ['B02', 'B03', 'B04', 'B08']
    #features = ['B04', 'B12']


    features = ['B8A', 'B11', 'B12', 'B05', 'B06', 'SCL'] # 'B8A', 'B11', 'B12', 'B05', 'B06']
    high_res_features = ['B02', 'B03', 'B04',  'B08']

    feature_names = high_res_features + features[:-1] + ['ndvi', 'ndwi', 'bsi']
    feature_names = feature_names +\
        list(map(lambda x: x + '_1', feature_names)) +\
        list(map(lambda x: x + '_2', feature_names))

    # (8a - 11) / (8a + 11) ['B8A', 'B11', 'B12' 'B02', 'B03', 'B04',  'B08', ndvi, ndwi 'B8A', 'B11', 'B12' 'B02', 'B03', 'B04',  'B08', ndvi, ndwi]

    #------------ getting data
    # pat_high, pat_low = prepare_data_patterns('data')
    # data =[np.random.randint(5110000, 5380000, size=50), np.random.randint(514000, 618000, size=50)]
    # res = get_data(data[0], data[1], ['B03'],  patterns=pat_high)

    # print(res)

    data = make_training_points_csv(filename='general.csv')



    # # _________ write data to shapefile


    # df = pd.DataFrame({'lat': presence_points[1], 'lon': presence_points[0]})
    # df['geometry'] = df.apply(lambda row: Point(row.lat,row.lon,0),axis=1)
    # df = df.drop(['lat', 'lon'], axis=1)
    # crs = {'init': 'epsg:32654'}

    # gdf = gpd.GeoDataFrame(df, geometry=df.geometry, crs=crs)
    # gdf.to_file('presence.shp')



    # z=get_data([5023621], [554192], ['B02'], normalize=False, patterns=DATA_PATTERNS10m)

    X = prepare_dataset(data[0], data[1], high_res_features, features, paths=['data', 'data2', 'data3']) #'data2', 'data3'])
    y = data[2]

    nan_maskX = np.isnan(X).any(axis=1)
    X = X[~nan_maskX]
    y = np.array(y)[~nan_maskX]


    # --------------- Betula - Salix classifier -------------------------

    # betula_salix_data = make_training_points_csv(filename='betula_salix.csv')

    # X_bs = prepare_dataset(betula_salix_data[0], betula_salix_data[1], high_res_features, features, paths=['data', 'data2', 'data3']) #'data2', 'data3'])
    # y_bs = betula_salix_data[2]

    # nan_maskX_bs = np.isnan(X_bs).any(axis=1)
    # X_bs = X_bs[~nan_maskX_bs]
    # y_bs = np.array(y_bs)[~nan_maskX_bs]

    # bsalix_mask = np.in1d(y_bs, np.array(['salix', 'betula']))
    # X_bs = X_bs[bsalix_mask]
    # y_bs = y_bs[bsalix_mask]

    # --------------- evergreen classifier -------------------------

    pl_data = make_training_points_csv(filename='evergreen.csv')

    X_pl = prepare_dataset(pl_data[0], pl_data[1], high_res_features, features, paths=['data', 'data2', 'data3']) #'data2', 'data3'])
    y_pl = pl_data[2]

    nan_maskX_pl = np.isnan(X_pl).any(axis=1)
    X_pl = X_pl[~nan_maskX_pl]
    y_pl = np.array(y_pl)[~nan_maskX_pl]



    # --------------- pinus classifier -------------------------

    pin_data = make_training_points_csv(filename='pinus.csv')

    X_pin = prepare_dataset(pin_data[0], pin_data[1], high_res_features, features, paths=['data', 'data2', 'data3']) #'data2', 'data3'])
    y_pin = pin_data[2]

    nan_maskX_pin = np.isnan(X_pin).any(axis=1)
    X_pin = X_pin[~nan_maskX_pin]
    y_pin = np.array(y_pin)[~nan_maskX_pin]


    #breakpoint()

    # from shapely.geometry import Point
    # import geopandas as gpd
    # df = pd.DataFrame({'lat': data[0], 'lon': data[1]})
    # df['geometry'] = df.apply(lambda row: Point(row.lon, row.lat, 0), axis=1)
    # df = df.drop(['lat', 'lon'], axis=1)
    # crs = {'init': 'epsg:32654'}
    # #df['data'] = data[2]res
    # df['X1'] = X[:, 1]
    # gdf = gpd.GeoDataFrame(df, geometry=df.geometry, crs=crs)
    # gdf.to_file('absence.shp')

    # pca = PCA(n_components=2)


    # ================== PCA of X =====================
    # coords = np.array(coords)
    # print(f"Length of coordinate array: {len(coords)}.")
    # for fn in range(15):
    #     print(f"Processing line {fn}.")
    #     # Get an index which we can use to re-spawn the oldest raindrop.
    #     y_ = (y==1) + (y == 0) * (np.array(z) <= fn)
    #     Xpr = pca.fit_transform(X[y_])
    #     # Update the scatter collection, with the new colors, sizes and positions.
    #     fig = plt.figure(figsize=(10, 10))
    #     ax1 = fig.add_subplot(211)
    #     ax2 = fig.add_subplot(212)
    #     ax2.set_aspect('equal', adjustable='box')

    #     n = sum(y==1)
    #     ax1.set_title(str(fn))
    #     ax1.scatter(Xpr[:n, 0], Xpr[:n, 1], alpha=0.8, color='b')
    #     ax1.scatter(Xpr[n:, 0], Xpr[n:, 1], alpha=0.8, color='r')
    #     #ax2.scatter(coords[y==1, 0], coords[y==1, 1], color='b')
    #     ax2.plot([info_absence.get(j)[1] for j in range(fn + 1) if info_absence.get(j) is not None],
    #              [info_absence.get(j)[0] for j in range(fn + 1) if info_absence.get(j) is not None], 'ro')
    #     plt.show()

    # for c, v in zip('rb', [0, 1]):
    #     plt.scatter(Xpr[y == v, 0], Xpr[y == v, 1], alpha=.8, color=c)

    # print(X[y==1, ...])
    # plt.show()

    # df = pd.DataFrame(X, columns=['r', 'g', 'b'])
    # df.loc[:, 'y'] = y
    # sns.pairplot(df, hue='y')
    # plt.show()



    # =================================================
    #feature_names = np.array(high_res_features + features + ['ndvi', 'ndwi'])
    # allowed = ['B02', 'B03', 'B08', 'B8A', 'B11', 'B12', 'ndwi']
    # mask = [True if name in allowed else False for name in feature_names]

    # print(f"Data matrix is bulit: {X.shape}. {X[:10]}")
    # breakpoint()
    # # sdf
    lab_enc = LabelEncoder()
    lab_enc.fit(y.tolist() + y_pl.tolist()+y_pin.tolist())
    print(lab_enc.classes_)



    y = lab_enc.transform(y)
    y_pl = lab_enc.transform(y_pl)
    y_pin = lab_enc.transform(y_pin)

    clf = RandomForestClassifier(n_estimators=500, random_state=10)
    #clf = LogisticRegression()
    #clf_rfecv = clf
    clf_rfecv = RFECV(clf, n_jobs=6, cv=5, verbose=0, scoring='balanced_accuracy')


    # clf_rfecv.fit(X, y)
    # scores = cross_val_score(clf_rfecv, X, y, cv=10, scoring='balanced_accuracy')
    # print(f"Final scores: {scores}.")
    # print(f"Mean accuracy score: {np.mean(scores)}.")


    plot_error_matrix(clf_rfecv, X, y, lab_enc, feature_names)


    # for j in range(1, 2 ** (len(feature_names))):
    #     selector = list(map(lambda x: bool(int(x)), list(format(j, f'#0{len(feature_names) + 2}b')[2:])))
    #     X_ = X[:, selector]
    #     scores = cross_val_score(clf, X_, y, cv=10, scoring='balanced_accuracy')
    #     if np.mean(scores) > 0.6:
    #         print(f"Features selected: {feature_names[selector]}")
    #         print(f"Accuracy score mean: {np.mean(scores)} +/- {np.std(scores)}.")
    #         print("=" * 50)
    #     else:
    #         print(f'Low score. {np.mean(scores)}')

    # y_probs = clf.predict_proba(X[:, mask])
    # for th in np.linspace(0, 1, 100):
    #     score = balanced_accuracy_score(y, (y_probs[:, 1]>th).astype(int))
    #     print(f"th: {th}; score = {score}")

    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    # ax.plot(X[y==0, 3], X[y==0, 2], 'b.' )
    # ax.plot(X[y==1, 3], X[y==1, 2], 'r.', alpha=0.7)
    # # ax.plot(*absence_points, 'r.')
    # # ax.plot(*presence_points, 'b.')
    # plt.show()
    # plt.savefig('output.png')





    # ----------------- Getting map ------------------------------------

    # y_predict = clf.predict(X)


    # print("Balanced score: ", balanced_accuracy_score(y, y_predict))
    # print("Accuracy score: ", accuracy_score(y, y_predict))


    # sakhalin
    left_up_sh = (48.38, 141.67)
    right_low_sh = [45.88, 143.80]

    lon_min, lat_max = convert_coords(*left_up_sh)
    lon_max, lat_min = convert_coords(*right_low_sh)

    cell_size = 100000
    xsize = ysize = 20


    n_cells = len(np.arange(lat_min, lat_max, cell_size)) * len(np.arange(lon_min, lon_max, cell_size))


    clf_bs = RandomForestClassifier(n_estimators=500, random_state=10)
    clf_rfecv_bs = RFECV(clf_bs, n_jobs=4, cv=5, verbose=0, scoring='balanced_accuracy')

    clf_pl = RandomForestClassifier(n_estimators=500, random_state=10)
    clf_rfecv_pl = RFECV(clf_pl, n_jobs=4, cv=5, verbose=0, scoring='balanced_accuracy')

    clf_pin = RandomForestClassifier(n_estimators=500, random_state=10)
    clf_rfecv_pin = RFECV(clf_pin, n_jobs=4, cv=5, verbose=0, scoring='balanced_accuracy')

    print(f"Total number of cells to be processed: {n_cells}.")

    for pth in [ ['data', 'data2', 'data3'],]:
        plot_error_matrix(clf_rfecv_pl, X_pl[:,:12*len(pth)], y_pl, lab_enc, feature_names)
        plot_error_matrix(clf_rfecv_pin, X_pin[:,:12*len(pth)], y_pin, lab_enc, feature_names)
        sys.exit(1)
        clf_rfecv.fit(X[:,:12*len(pth)], y)
        #clf_rfecv_bs.fit(X_bs[:,:9*len(pth)], y_bs)
        clf_rfecv_pl.fit(X_pl[:,:12*len(pth)], y_pl)
        clf_rfecv_pin.fit(X_pin[:,:12*len(pth)], y_pin)
        for lat in np.arange(lat_min, lat_max, cell_size):
            for lon in np.arange(lon_min, lon_max, cell_size):

                lats_sh, lons_sh = np.meshgrid(
                    np.arange(lat, lat + cell_size, ysize),
                    np.arange(lon, lon + cell_size, xsize)
                )
                MAPX = prepare_dataset(
                    lats_sh.ravel(),
                    lons_sh.ravel(),
                    high_res_features,
                    features,
                    paths=pth,
                    filter_by_scl=[]
                )
                nan_mask = np.isnan(MAPX).any(axis=1)
                y_predict = np.full_like(MAPX[:, 0], np.nan)
                if not all(MAPX[~nan_mask].shape):
                    continue
                y_predict[~nan_mask] = clf_rfecv.predict(MAPX[~nan_mask])

                # apply to broadleaf
                # bleaf_mask = y_predict == lab_enc.transform(['broadleaf'])[0]
                # if any(bleaf_mask):
                #     y_predict[bleaf_mask] = clf_rfecv_bs.predict(MAPX[bleaf_mask])


                # apply to evergreen
                egreen_mask = y_predict == lab_enc.transform(['evergreen'])[0]
                if any(egreen_mask):
                    y_predict[egreen_mask] = clf_rfecv_pl.predict(MAPX[egreen_mask])

                # apply to pinus
                pinus_mask = y_predict == lab_enc.transform(['pinus'])[0]
                if any(pinus_mask):
                    y_predict[pinus_mask] = clf_rfecv_pin.predict(MAPX[pinus_mask])


                xmin = lons_sh.min()
                xres = xsize
                ymax = lats_sh.max()
                yres = -ysize
                GeoT = (xmin, xres, 0, ymax, 0, yres)

                y_predict[np.isnan(y_predict)] = 254
                y_predict = y_predict.astype(np.uint8)

                save_as_tiff(y_predict, *lats_sh.shape, GeoT,
                            fname=f'{"_".join(pth)}_prediction_{lat}_{lon}')
    #save_as_tiff(MAPX[:,1], *lats_sh.shape, GeoT, fname='first_mapx')



    print("Prediction: ", Counter(y_predict))
    print("Training dataset: ", Counter(y))
