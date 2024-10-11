import os.path
from glob import glob
from typing import Union, Optional

import cv2
import geopandas as gpd
import numpy as np
import shapely
from osgeo import gdal


def create_label(png_path):
    """
    根据像素的个数
    :param png_path:
    :return:
    """


def tif2png(file_path, save_path):
    """
    将tif形式的标注图像转换为png形式的标注
    :param file_path:
    :param save_path:
    :return:
    """
    if os.path.isdir(file_path):
        file_path = glob(file_path + '/*')
    else:
        file_path = [file_path]
    for i in range(len(file_path)):
        file = file_path[i]
        filename = os.path.split(file)[-1]
        name, ext = os.path.splitext(filename)
        assert ext == '.tif'
        save_name = save_path + '/' + name + '.png'
        tif_data = gdal.Open(file)
        png_driver = gdal.GetDriverByName('PNG')
        png_driver.CreateCopy(save_name, tif_data)


def shp2tif(shp_path, tif_path, label_path):
    """
    将标注的shp文件转换为tif格式的遥感图像标注
    :param shp_path: shp文件的路径
    :param tif_path: 标注对应的tif遥感图像的路径
    :param label_path: 转换为tif格式的标注图像的保存路径
    """
    ori_tif = gdal.Open(tif_path)
    ori_proj = ori_tif.GetProjection()
    ori_geotrans = ori_tif.GetGeoTransform()
    ori_width = ori_tif.RasterXSize
    ori_height = ori_tif.RasterYSize
    # 释放内存
    ori_tif = None
    shp_driver = gdal.GetDriverByName('ESRI Shapefile')
    ds = shp_driver.Open(shp_path, 1)
    # 获取图层文件对象
    layer = ds.GetLayer()

    tif_driver = gdal.GetDriverByName('GTiff')
    tif_data = tif_driver.Create(label_path, ori_width, ori_height, bands=1, eType=gdal.GDT_Byte)
    # 写入投影信息
    tif_data.SetProjection(ori_proj)
    # 写入地理空间变换信息
    tif_data.SetGeoTransform(ori_geotrans)
    band = tif_data.GetRasterBand(1)
    # 背景值设置为0
    band.SetNoDataValue(0)
    # 清空数据缓存
    band.FlushCache()
    # gdal.RasterizeLayer(tif_data, [1], layer, burn_values=[1], options=[f"ATTRIBUTE={attribute}"])
    gdal.RasterizeLayer(tif_data, [1], layer, burn_values=[1])
    tif_data = None
    del tif_data, layer


def split2multi_shp(file_path, conditions: dict = None):
    """
    将一个文件中（包含多个多边形）的多个多边形区域分成多个单独的shp文件，每个shp文件都是单个多边形。
    （之所以这么做是因为目标区域太大，没有存储空间，需要分成多个部分后在合并）

    使用场景：将一个县的shp拆分成多个镇的shp；将一个镇的shp拆分成多个村的shp

    :param conditions: 筛选条件
    :param file_path: 源文件路径
    :return:
    """
    gdf = gpd.read_file(file_path)
    filename = os.path.splitext(os.path.split(file_path)[-1])[0]
    if conditions is not None:
        for k, v in conditions.items():
            gdf = gdf[gdf[k] == v]
    # gdf = gdf[gdf['type'] == 'boundary']
    for i in range(len(gdf)):
        sub_gdf = gdf.iloc[i:i + 1]
        if sub_gdf.get('name') is not None:
            name_str = sub_gdf['name'].values[0]
        elif sub_gdf.get('NAME') is not None:
            name_str = sub_gdf['NAME'].values[0]
        else:
            name_str = str(i)
        sub_name = filename + '_' + name_str
        save_path = os.path.split(file_path)[0] + '/' + sub_name + '.shp'
        sub_gdf.to_file(save_path)


def multi_shp2one_shp(shp_paths: list, save_path=None):
    """
    将多个shp文件合并成单个shp文件

    使用场景：将多个村的shp合并成一个镇的shp；将多个镇的shp合并成一个村的shp

    :param save_path: 合并后的shp文件保存路径
    :param shp_paths: 需要合并的shp文件的路径列表
    :return:
    """
    one_gpd = None
    for p in shp_paths:
        sub_gpd = gpd.read_file(p)
        if one_gpd is None:
            one_gpd = sub_gpd
        else:
            sub_gpd.to_crs(one_gpd.crs)
            one_gpd.merge(sub_gpd)
    if save_path is None:
        save_path = os.path.split(shp_paths[0])[0] + '/output.shp'
    one_gpd.to_file(save_path)


def add_field(data, field_name, field_value):
    pass


def ndarray2shp(mask_data: np.ndarray, shp_path: str, tif_path: str, clip_shp: str, clip_condition: dict):
    in_raster = gdal.Open(tif_path)
    geo_trans = in_raster.GetGeoTransform()
    contours, _ = cv2.findContours(mask_data, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if clip_shp is not None:
        points = []
        groups = []
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if 1000 < area < 1000000:
                for i, point in enumerate(contour):
                    x_col = geo_trans[0] + geo_trans[1] * (float(point[0, 0])) + geo_trans[2] * (float(point[0, 1]))
                    y_row = geo_trans[3] + geo_trans[4] * (float(point[0, 0])) + geo_trans[5] * (float(point[0, 1]))
                    p = shapely.Point(x_col, y_row)
                    points.append(p)
                    groups.append(idx)
        gdf = gpd.GeoDataFrame(crs=in_raster.GetProjection(), geometry=points)
        gdf['groups'] = groups
        clip_area_gdf = gpd.read_file(clip_shp).to_crs(gdf.crs)
        if clip_condition is not None and len(clip_condition) > 0:
            for k, v in clip_condition.items():
                clip_area_gdf = clip_area_gdf[clip_area_gdf[k] == v]
        gdf = gdf.sjoin(clip_area_gdf)
        grouped_data = gdf.groupby('groups')
        new_polygons = []
        for data in grouped_data:
            points = [(d.x, d.y) for d in data[1].geometry]
            if len(points) < 4:
                continue
            new_polygons.append(shapely.Polygon(points))
        gdf = gpd.GeoDataFrame(geometry=new_polygons, crs=gdf.crs)
        gdf['Area'] = gdf.area
    else:
        polygons = []
        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if 1000 < area < 1000000:
                points = []
                for i, point in enumerate(contour):
                    x_col = geo_trans[0] + geo_trans[1] * (float(point[0, 0])) + geo_trans[2] * (float(point[0, 1]))
                    y_row = geo_trans[3] + geo_trans[4] * (float(point[0, 0])) + geo_trans[5] * (float(point[0, 1]))
                    points.append((x_col, y_row))
                polygons.append(shapely.Polygon(points))
        gdf = gpd.GeoDataFrame(crs=in_raster.GetProjection(), geometry=polygons)
    gdf.to_file(shp_path)


def analyse(shp1: Union[str, gpd.GeoDataFrame],
            shp2: Union[str, gpd.GeoDataFrame],
            filter1: Optional[dict] = None,
            filter2: Optional[dict] = None):
    """
    比较两个区域的面积

    :param filter2: shp2的过滤条件
    :param filter1: shp1的过滤条件
    :param shp2: shp2文件路径或者读取后的GeoDataFrame对象（被对比的对象）
    :param shp1: shp1文件路径或者读取后的GeoDataFrame对象（需要对比的对象）
    :return:
    """
    if type(shp1) == str:
        shp1 = gpd.read_file(shp1)
    if type(shp2) == str:
        shp2 = gpd.read_file(shp2)
    if filter1:
        for k, v in filter1.items():
            shp1 = shp1[shp1[k] == v]
    if filter2:
        for k, v in filter2.items():
            shp2 = shp2[shp2[k] == v]

    shp1_area = sum(shp1.area)
    shp2_area = sum(shp2.area)

    diff = shp2_area - shp1_area

    percent = (abs(diff) / shp1_area) * 100

    flag = 'higher' if diff > 0 else 'lower'

    return shp1_area, abs(diff), flag, percent


def read_tif(img_path):
    tif_data = gdal.Open(img_path)
    arr = tif_data.ReadAsArray()
    type = tif_data.GetRasterBand(1).DataType
    projection = tif_data.GetProjection()
    geo_trans = tif_data.GetGeoTransform()
    del tif_data
    return arr, type, projection, geo_trans


def write_tif(data: np.ndarray, type, geo_trans, projection, save_path):
    """
    目前只是针对单通道的标签图
    :param data:
    :param type:
    :param geo_trans:
    :param projection:
    :param save_path:
    :return:
    """
    tif_driver = gdal.GetDriverByName('GTiff')
    if len(data.shape) == 2:
        data = data[None, :, :]
    channels, height, width = data.shape
    new_tif = tif_driver.Create(save_path, width, height, channels, type)
    new_tif.SetProjection(projection)
    new_tif.SetGeoTransform(geo_trans)
    for i in range(channels):
        new_tif.GetRasterBand(i + 1).WriteArray(data[i])
    del new_tif


def fix_tif(tif_path):
    """
    修正标签值
    :param tif_path:
    :return:
    """
    arr, type, projection, geo_trans = read_tif(tif_path)
    arr_copy = arr.copy()
    # arr[arr_copy == 0] = 1
    # arr[arr_copy == 1] = 1
    arr[arr_copy == 255] = 1
    save_path, filename = os.path.split(tif_path)
    name, ext = os.path.splitext(filename)
    save_path = save_path + '/' + name + '_fix' + ext
    write_tif(arr, type, geo_trans, projection, save_path)


def batch_fix_tif(file_folder):
    """
    批量修正标签值
    :param file_folder:
    :return:
    """
    file_paths = glob(file_folder + '/*')
    for path in file_paths:
        fix_tif(path)


def batch_rename(file_path, identity_str):
    """
    批量修改文件名
    :param file_path:
    :param identity_str:
    :return:
    """
    paths = glob(file_path + '/*')
    for i, p in enumerate(paths):
        ori_path, ori_name = os.path.split(p)
        name, ext = os.path.splitext(ori_name)
        new_path = os.path.join(ori_path, identity_str + str(i + 1) + ext)
        os.rename(p, new_path)


if __name__ == '__main__':
    # shp_lists = [
    #     'D:/datasets/Cropland_Identity/cropland_identity_datasource/predicts/pseudo_color_prediction/河坝镇.shp',
    #     'D:/datasets/Cropland_Identity/cropland_identity_datasource/predicts/pseudo_color_prediction/金盆镇.shp',
    #     'D:/datasets/Cropland_Identity/cropland_identity_datasource/predicts/pseudo_color_prediction/柳林洲街道.shp',
    #     'D:/datasets/Cropland_Identity/cropland_identity_datasource/predicts/pseudo_color_prediction/营田镇.shp',
    #     'D:/datasets/Cropland_Identity/cropland_identity_datasource/predicts/pseudo_color_prediction/钱粮湖镇.shp',
    #     'D:/datasets/Cropland_Identity/cropland_identity_datasource/predicts/pseudo_color_prediction/凤凰乡.shp'
    # ]
    # filter_list = [
    #     {'ZLDWMC': '芸洲子村'},
    #     {'ZLDWMC': '有成村'},
    #     {'ZLDWMC': '二洲子村'},
    #     {'ZLDWMC': '余家坪社区'},
    #     {'ZLDWMC': '三角闸村'},
    #     {'ZLDWMC': '磊石村'}
    # ]
    # fid_name = ['芸洲子村', '有成村', '二洲子村', '余家坪社区', '三角闸村', '磊石村']
    # flag_list = []
    # percent_list = []
    # diff_list = []
    # contrast_shp = 'D:/datasets/Cropland_Identity/6个村归档/6个村的耕地.shp'
    # contrast_gdf = gpd.read_file(contrast_shp)
    # ori_list = []
    # for shp, fit in zip(shp_lists, filter_list):
    #     ori_area, diff, flag, percent = analyse(contrast_gdf, shp, filter1=fit)
    #     diff_list.append(diff)
    #     flag_list.append(flag)
    #     ori_list.append(ori_area)
    #     percent_list.append(percent)
    # df = pd.DataFrame({'fid_name': fid_name, 'ori_area': ori_list, 'diff': diff_list, 'flag': flag_list, 'percent': percent_list})
    # print(df)
    # df.to_csv('analyse_result.csv', index=False)
    # df.to_json('analyse_result.json')

    # split2multi_shp('C:/Users/simon/Desktop/归档(1)/东安县.shp')

    # shp2tif('D:/datasets/湘潭县/湘潭县_白石镇.shp', 'd:/test_1.tif', None)

    arr, type, projection, geo_trans = read_tif('D:/datasets/xaingtan/dataset/part4/labels/part4_image_6.tif')
    print(np.unique(arr))

    # batch_rename('D:/datasets/xaingtan/dataset/part4/images', 'part4_image_')
    # batch_rename('D:/datasets/xaingtan/dataset/part4/labels', 'part4_image_')
    # batch_rename('D:/datasets/xaingtan/dataset/part5/images', 'part5_image_')
    # batch_rename('D:/datasets/xaingtan/dataset/part5/labels', 'part5_image_')
    # batch_rename('D:/datasets/xaingtan/dataset/part6/images', 'part6_image_')
    # batch_rename('D:/datasets/xaingtan/dataset/part6/labels', 'part6_image_')
    # batch_rename('D:/datasets/xaingtan/dataset/part7/images', 'part7_image_')
    # batch_rename('D:/datasets/xaingtan/dataset/part7/labels', 'part7_image_')
    # batch_rename('D:/datasets/xaingtan/dataset/part8/images', 'part8_image_')
    # batch_rename('D:/datasets/xaingtan/dataset/part8/labels', 'part8_image_')
    # batch_rename('D:/datasets/xaingtan/dataset/part9/images', 'part9_image_')
    # batch_rename('D:/datasets/xaingtan/dataset/part9/labels', 'part9_image_')

    # fix_tif('D:/datasets/xaingtan/2/area_0928_label.tif')

    # batch_fix_tif('D:/datasets/xaingtan/dataset/part4/labels')
    # batch_fix_tif('D:/datasets/xaingtan/dataset/part5/labels')
    # batch_fix_tif('D:/datasets/xaingtan/dataset/part6/labels')
    # batch_fix_tif('D:/datasets/xaingtan/dataset/part7/labels')
    # batch_fix_tif('D:/datasets/xaingtan/dataset/part8/labels')
    # batch_fix_tif('D:/datasets/xaingtan/dataset/part9/labels')
