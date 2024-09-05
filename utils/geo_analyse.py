from typing import Union, Optional

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from osgeo import gdal


def ndarray2shp(mask_data: np.ndarray, shp_path: str, tif_path: str, clip_shp: str, clip_condition: dict):
    in_raster = gdal.Open(tif_path)
    geo_trans = in_raster.GetGeoTransform()
    contours, _ = cv2.findContours(mask_data, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
    if clip_shp is not None:
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
    gdf.to_file(shp_path)


# def data_to_shp(mask_data: np.ndarray, tif_path, shp_path, clip_target: str = None, conditions: dict = None):
#     """
#     将预测得到的分割图转换为shp文件中的多边形
#     :param mask_data: 预测得到的分割数据
#     :param tif_path: 需要预测的输入tif文件
#     :param shp_path: 输出shp的保存路径
#     :param clip_target: 包含了tif_path所代表区域的父区域（如果提供了的话）的shp图层文件，其中包含了多个和tif_path相似的区域
#     :param conditions: 用于选择clip_target中的指定区域，tif_path预测时是一个规则的矩形图片，但需要的往往是一个不规则的多边形区域，
#                        该多边形区域是tif_path中的一部分，因此需要对预测结果进行裁剪，裁剪到需要的不规则的多边形区域，conditions参
#                        数用于从clip_target中筛选出需要的多边形区域作为裁剪的对照
#     :return:
#     """
#     ogr.RegisterAll()
#     # gdal.SetConfigOption('GDAL_FILENAME_IS_UTF8', 'NO')
#     # gdal.SetConfigOption('SHAPE_ENCODING', 'CP936')
#     driver = ogr.GetDriverByName('ESRI Shapefile')
#     temp_shp_path = './temp.shp'
#     if os.path.exists(shp_path):
#         driver.DeleteDataSource(shp_path)
#     if os.path.exists(temp_shp_path):
#         driver.DeleteDataSource(temp_shp_path)
#     in_raster = gdal.Open(tif_path)
#     prj = osr.SpatialReference()
#     prj.ImportFromWkt(in_raster.GetProjection())
#     geo_trans = in_raster.GetGeoTransform()
#
#     temp_polygon = driver.CreateDataSource(temp_shp_path)
#     temp_layer = temp_polygon.CreateLayer(temp_shp_path[:-4], srs=prj, geom_type=ogr.wkbPolygon, options=[])
#
#     field_id = ogr.FieldDefn('FieldId', ogr.OFTInteger)
#     temp_layer.CreateField(field_id)
#     field_area = ogr.FieldDefn('area', ogr.OFTReal)
#     field_area.SetWidth(32)
#     field_area.SetPrecision(15)
#     temp_layer.CreateField(field_area)
#     defn = temp_layer.GetLayerDefn()
#
#     gardens = ogr.Geometry(ogr.wkbMultiPolygon)
#     contours, _ = cv2.findContours(mask_data, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     i = 0
#     for contour in reversed(contours):
#         area = cv2.contourArea(contour)
#         box1 = ogr.Geometry(ogr.wkbLinearRing)
#         i += 1
#         init_x = 0.
#         init_y = 0.
#         if area >= 10:
#             for idx, point in enumerate(contour):
#                 x_col = geo_trans[0] + geo_trans[1] * (float(point[0, 0])) + geo_trans[2] * (float(point[0, 1]))
#                 y_row = geo_trans[3] + geo_trans[4] * (float(point[0, 0])) + geo_trans[5] * (float(point[0, 1]))
#                 if idx == 0:
#                     init_x = x_col
#                     init_y = y_row
#                 box1.AddPoint(x_col, y_row)
#             box1.AddPoint(init_x, init_y)
#             garden_item = ogr.Geometry(ogr.wkbPolygon)
#             garden_item.AddGeometry(box1)
#             garden_item.CloseRings()
#             feature_triangle = ogr.Feature(defn)
#             feature_triangle.SetField('FieldId', i)
#             feature_triangle.SetField('area', garden_item.GetArea())
#             feature_triangle.SetGeometry(garden_item)
#             gardens.AddGeometry(garden_item)
#             # temp_layer.CreateFeature(feature_triangle)
#     gardens.CloseRings()
#     feature = ogr.Feature(defn)
#     feature.SetGeometry(gardens)
#     temp_layer.CreateFeature(feature)
#     temp_polygon.SyncToDisk()
#     temp_polygon.Destroy()
#
#     if clip_target is not None:
#         clip_raster_by_shp(temp_shp_path, clip_target, conditions)
#         # temp_source = driver.Open(temp_shp_path)
#         # source_layer = temp_source.GetLayer()
#         #
#         # tgt = driver.Open(clip_target)
#         #
#         # # tgt.SetSpatialRef(temp_source.GetSpatialRef())
#         #
#         # target_layer = tgt.GetLayer()
#         # # target_layer.SetActiveSRS(0, source_layer.GetSpatialRef())
#         # for k, v in conditions.items():
#         #     # ogr.Layer.SetSpatialFilter(target_layer, )
#         #     target_layer.SetAttributeFilter(f'{k}={v}')
#         #     # target_layer.SetFeature(target_layer[0])
#         # polygon = driver.CreateDataSource(shp_path)
#         # poly_layer = polygon.CreateLayer(shp_path[:-4], source_layer.GetSpatialRef(), source_layer.GetGeomType())
#         # source_layer.SetSpatialFilter(target_layer[0].geometry())
#         # source_layer.Clip(target_layer, poly_layer)
#         # polygon.FlushCache()
#         # temp_source.Destroy()
#         # tgt.Destroy()
#         # polygon.SyncToDisk()
#         # 下面这行代码不能漏
#         # polygon.Destroy()
#
#
# def clip_raster_by_shp(source, target, conditions: dict = None, output_folder=None):
#     """
#     根据target文件中的多边形区域形状裁剪source文件的整个区域，使其裁剪后与target的多边形区域相同，
#     其中target为某个区域的一个整体的多边形
#     :param source: 需要裁剪的shp文件路径
#     :param target: 目标shp文件路径
#     :param conditions: 一个区域筛选条件，用于筛选target中的指定区域，通常用于target是一个包含多个离散区域（如村庄等）的情况
#     :param output_folder: 裁剪后的文件保存路径，默认为None，即直接替换source文件
#     """
#     src = gpd.read_file(source)
#     tgt = gpd.read_file(target).to_crs(src.crs)
#     if output_folder is None:
#         output_folder = source
#     if conditions is not None:
#         out_path, out_ext = os.path.splitext(output_folder)
#         for key, value in conditions.items():
#             out_path = out_path + '_' + key + '_' + str(value)
#         output_folder = out_path + out_ext
#     for k, v in conditions.items():
#         tgt = tgt[tgt[k] == v]
#     clipped_src = src.clip(tgt)
#     collection_type = clipped_src[clipped_src['geometry'].type == 'GeometryCollection']
#     for i in range(len(collection_type)):
#         geometry_item = collection_type['geometry']
#         str_geo = str(geometry_item.iloc[2])
#         new_geometry = str_geo.split(', LINESTRING')[0] + ')'
#         collection_type.iloc[i]['geometry'] = shapely.from_wkt(new_geometry)
#     clipped_src.to_file(output_folder)
#     return output_folder
#
#
# def compare_area(shp1, shp2):
#     """
#     获取两个shp文件中多边形的面积，并进行比较分析，两个shp属于同一个区域
#     :param shp1: 第一个shp文件的路径（源，被比较，地块识别场景中为提供的）
#     :param shp2: 第二个shp文件的路径（目标，去比较，地块识别场景中为自己预测的）
#     :return: 统计分析
#     """
#     gdf_1 = gpd.read_file(shp1)
#     gdf_2 = gpd.read_file(shp2).to_crs(gdf_1.crs)
#     area_1 = sum(gdf_1.area)
#     area_2 = sum(gdf_2.area)
#     return area_1, area_2
#
#
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


if __name__ == '__main__':
    shp_lists = [
        'D:/datasets/Cropland_Identity/cropland_identity_datasource/predicts/pseudo_color_prediction/河坝镇.shp',
        'D:/datasets/Cropland_Identity/cropland_identity_datasource/predicts/pseudo_color_prediction/金盆镇.shp',
        'D:/datasets/Cropland_Identity/cropland_identity_datasource/predicts/pseudo_color_prediction/柳林洲街道.shp',
        'D:/datasets/Cropland_Identity/cropland_identity_datasource/predicts/pseudo_color_prediction/营田镇.shp',
        'D:/datasets/Cropland_Identity/cropland_identity_datasource/predicts/pseudo_color_prediction/钱粮湖镇.shp',
        'D:/datasets/Cropland_Identity/cropland_identity_datasource/predicts/pseudo_color_prediction/凤凰乡.shp'
    ]
    filter_list = [
        {'ZLDWMC': '芸洲子村'},
        {'ZLDWMC': '有成村'},
        {'ZLDWMC': '二洲子村'},
        {'ZLDWMC': '余家坪社区'},
        {'ZLDWMC': '三角闸村'},
        {'ZLDWMC': '磊石村'}
    ]
    fid_name = ['芸洲子村', '有成村', '二洲子村', '余家坪社区', '三角闸村', '磊石村']
    flag_list = []
    percent_list = []
    diff_list = []
    contrast_shp = 'D:/datasets/Cropland_Identity/6个村归档/6个村的耕地.shp'
    contrast_gdf = gpd.read_file(contrast_shp)
    ori_list = []
    for shp, fit in zip(shp_lists, filter_list):
        ori_area, diff, flag, percent = analyse(contrast_gdf, shp, filter1=fit)
        diff_list.append(diff)
        flag_list.append(flag)
        ori_list.append(ori_area)
        percent_list.append(percent)
    df = pd.DataFrame({'fid_name': fid_name, 'ori_area': ori_list, 'diff': diff_list, 'flag': flag_list, 'percent': percent_list})
    print(df)
    df.to_csv('analyse_result.csv', index=False)
    df.to_json('analyse_result.json')
