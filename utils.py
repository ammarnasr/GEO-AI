import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
import geopandas as gpd
import shapely.geometry
import senHub_utils
import os
import rasterio

def read_data(filepath):
    df = pd.read_csv(filepath)
    lats = df['Lat'].values
    lons = df['Lon'].values
    points = []
    country = []
    for i in range(len(lats)):
        points.append(shapely.geometry.Point(lons[i], lats[i]))
        lat = lats[i]
        if lat > 14.0 and lat < 15.0:
            c = 'Sudan'
        elif lat > 32.0 and lat < 33.0:
            c = 'Iran'
        elif lat > 34.0 and lat < 35.0:
            c = 'Afghanistan'
        else:
            c = 'Unknown'
        country.append(c)
    gdf = gpd.GeoDataFrame(df, geometry=points)
    gdf.crs = {'init': 'epsg:4326'}
    gdf = gdf.to_crs(crs='EPSG:4326')
    gdf['Country'] = country
    return gdf

def read_train_data():
    filepath = 'train.csv'
    return read_data(filepath)

def read_test_data():
    filepath = 'test.csv'
    return read_data(filepath)

def shapely_point(lat, lon):
    return shapely.geometry.Point(lon, lat)

def shapely_polygon_from_bounds(bounds):
    return shapely.geometry.Polygon([[bounds[1], bounds[0]], [bounds[3], bounds[0]], [bounds[3], bounds[2]], [bounds[1], bounds[2]]])

def gdf_center(gdf):
    centers = gdf.centroid
    x_centers = [p.x for p in centers]
    y_centers = [p.y for p in centers]
    x_mean = sum(x_centers) / len(x_centers)
    y_mean = sum(y_centers) / len(y_centers)
    p_mean = shapely_point(y_mean, x_mean)
    return p_mean

def gdf_square_map(gdf, m=None, zoom_start=12):
    if len(gdf) == 0:
        return m
    bounds = gdf.total_bounds
    polygon = shapely_polygon_from_bounds(bounds)
    if m is not None:
        folium.Polygon(polygon.exterior.coords, color='red', name = 'bounds').add_to(m)
    else:
        p_center = gdf_center(gdf)
        m = folium.Map(location=[p_center.y, p_center.x], zoom_start=zoom_start)
        folium.Polygon(polygon.exterior.coords, color='red', name = 'bounds').add_to(m)
    return m

def add_esri_satellite_layer(m):
    folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = False,
        control = True).add_to(m)
    return m

def gdf_total_polygon(gdf):
    total_bounds = gdf.total_bounds
    total_polygon = shapely.geometry.box(*total_bounds, ccw=True)
    return total_polygon

def get_available_dates_from_sentinelhub(polygon, year='2023'):
    bounds = senHub_utils.get_bounds_of_polygon(polygon)
    token = senHub_utils.get_sentinelhub_api_token()
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'
    dates = senHub_utils.get_available_dates_from_sentinelhub(bounds, token, start_date, end_date)
    return dates

def get_dates_with_available_images_for_gdf_from_sentinelhub(gdf, year):
    if len(gdf) > 0:
        total_polygon = gdf_total_polygon(gdf)
        dates = get_available_dates_from_sentinelhub(total_polygon, year=year)
        return dates
    else:
        return []

def download_gdf_image_from_sentinelhub(gdf, date, evalscript, location, row=None):
    if row is not None:
        gdf = gdf.iloc[[row]]
    total_polygon = gdf_total_polygon(gdf)
    img, final_dir = senHub_utils.get_any_image_from_sentinelhub(total_polygon, date, evalscript, location)
    return img, final_dir

def get_evalscripts_info():
    senHub_utils.get_sentinelhub_api_evalscript(info=True)

def get_available_data_dataframe():
    satellite_images_dir = './data/satellite_images'
    downloaded_dates = os.listdir(satellite_images_dir)
    #remove dates if they are not in YYYY-MM-DD format
    downloaded_dates = [date for date in downloaded_dates if len(date) == 10]
    downloaded_dates = sorted(downloaded_dates)
    downloaded_data_dict = {
        'date': [],
        'location': [],
        'evalscript': [],
        'identifier': [],
        'file_name': []
    }
    for date in downloaded_dates:
        date_dir = os.path.join(satellite_images_dir, date)
        locations = os.listdir(date_dir)
        for location in locations:
            location_dir = os.path.join(date_dir, location)
            evalscripts = os.listdir(location_dir)
            for evalscript in evalscripts:
                evalscript_dir = os.path.join(location_dir, evalscript)
                identifiers = os.listdir(evalscript_dir)
                for identifier in identifiers:
                    identifier_dir = os.path.join(evalscript_dir, identifier)
                    file_names = os.listdir(identifier_dir)
                    for file_name in file_names:
                        if file_name.startswith('response'):
                            downloaded_data_dict['date'].append(date)
                            downloaded_data_dict['location'].append(location)
                            downloaded_data_dict['evalscript'].append(evalscript)
                            downloaded_data_dict['identifier'].append(identifier)
                            downloaded_data_dict['file_name'].append(file_name)
    downloaded_data = pd.DataFrame(downloaded_data_dict)
    return downloaded_data

def get_image_path_from_row(row):
    date = row['date']
    location = row['location']
    evalscript = row['evalscript']
    identifier = row['identifier']
    file_name = row['file_name']
    image_path = f'./data/satellite_images/{date}/{location}/{evalscript}/{identifier}/{file_name}'
    return image_path

def read_image_with_rasterio(image_path):
    img = rasterio.open(image_path)
    return img
