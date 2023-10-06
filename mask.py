import os
import utils
import folium
import joblib
import rasterio
import numpy as np
import pandas as pd
import multiprocessing
import geopandas as gpd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from multiprocessing import Pool



bands_all = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08','B8A', 'B09', 'B10', 'B11', 'B12']
bands_fcover = ['FCOVER']

def get_gdf_deviding_vals(gdf):
    lats = gdf['Lat'].values
    lons = gdf['Lon'].values
    total_bounds = gdf.total_bounds
    lats_min  = total_bounds[1]
    lats_max  = total_bounds[3]
    lats_deviding_val = lats_min + (lats_max-lats_min)/2
    lons_min  = total_bounds[0]
    lons_max  = total_bounds[2]
    lons_deviding_val = lons_min + (lons_max-lons_min)/2
    d = {
        'lats_min': lats_min,
        'lats_max': lats_max,
        'lats_deviding_val': lats_deviding_val,
        'lons_min': lons_min,
        'lons_max': lons_max,
        'lons_deviding_val': lons_deviding_val
    }
    return d

def get_north_west_gdf(gdf):
    d = get_gdf_deviding_vals(gdf)
    lats_deviding_val = d['lats_deviding_val']
    lons_deviding_val = d['lons_deviding_val']
    gdf = gdf[gdf['Lat'] > lats_deviding_val]
    gdf = gdf[gdf['Lon'] < lons_deviding_val]
    return gdf

def get_south_east_gdf(gdf):
    d = get_gdf_deviding_vals(gdf)
    lats_deviding_val = d['lats_deviding_val']
    lons_deviding_val = d['lons_deviding_val']
    gdf = gdf[gdf['Lat'] < lats_deviding_val]
    gdf = gdf[gdf['Lon'] > lons_deviding_val]
    return gdf

def get_north_east_gdf(gdf):
    d = get_gdf_deviding_vals(gdf)
    lats_deviding_val = d['lats_deviding_val']
    lons_deviding_val = d['lons_deviding_val']
    gdf = gdf[gdf['Lat'] > lats_deviding_val]
    gdf = gdf[gdf['Lon'] > lons_deviding_val]
    return gdf

def get_south_west_gdf(gdf):
    d = get_gdf_deviding_vals(gdf)
    lats_deviding_val = d['lats_deviding_val']
    lons_deviding_val = d['lons_deviding_val']
    gdf = gdf[gdf['Lat'] < lats_deviding_val]
    gdf = gdf[gdf['Lon'] < lons_deviding_val]
    return gdf

def donwload_images_for_all_scriprs(gdf, date, country):
    evalscript_clp = 'CLP'
    evalscript_true_color = 'TRUECOLOR'
    evalscript_all = 'ALL'
    evalscript_fcover = 'FCOVER'
    clp_image, clp_dir = utils.download_gdf_image_from_sentinelhub(gdf, date, evalscript_clp, country)
    true_color_image, true_color_dir = utils.download_gdf_image_from_sentinelhub(gdf, date, evalscript_true_color, country)
    all_image, all_dir = utils.download_gdf_image_from_sentinelhub(gdf, date, evalscript_all, country)
    fcover_image, fcover_dir = utils.download_gdf_image_from_sentinelhub(gdf, date, evalscript_fcover, country)
    d = {
        'clp_image': clp_image,
        'clp_dir': clp_dir,
        'true_color_image': true_color_image,
        'true_color_dir': true_color_dir,
        'all_image': all_image,
        'all_dir': all_dir,
        'fcover_image': fcover_image,
        'fcover_dir': fcover_dir
    }
    return d

def rastrio_img(df, index):
    row = df.iloc[index]
    img_path = utils.get_image_path_from_row(row)
    img = utils.read_image_with_rasterio(img_path)
    return img

def get_lats_lons_from_raster(src):
    band1 = src.read(1)
    height = band1.shape[0]
    width = band1.shape[1]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(src.transform, rows, cols)
    lons= np.array(xs)
    lats = np.array(ys)
    return lats, lons

def gdf_from_raster(src, bands_names=None):
    lats, lons = get_lats_lons_from_raster(src)
    img_array = np.array(src.read())
    if bands_names is None:
        bands_names = [f'B{i}' for i in range(src.count)]
    lats = lats.flatten()
    lons = lons.flatten()
    img_array = img_array.reshape((img_array.shape[0], img_array.shape[1]*img_array.shape[2]))
    df = pd.DataFrame(img_array.T, columns=bands_names)
    df['Lat'] = lats
    df['Lon'] = lons
    geom = []
    for i in tqdm(range(len(df))):
        p = utils.shapely_point(lats[i], lons[i]) 
        geom.append(p)
    gdf = gpd.GeoDataFrame(df, geometry=geom)
    gdf.crs = src.crs
    gdf = gdf.to_crs('EPSG:4326')
    gdf = gdf.reset_index(drop=True)
    return gdf

def get_gdf_from_row(df, row, bands_names=None):
    src = rastrio_img(df, row)
    gdf = gdf_from_raster(src, bands_names)
    return gdf

def get_processed_masked_src_gdf(gdf, masking_dict):
    eq_points_indicies = masking_dict['eq_points_indicies']
    eq_points_targets = masking_dict['eq_points_targets']
    gdf = gdf.iloc[eq_points_indicies]
    gdf = gdf.reset_index(drop=True)
    gdf['Target'] = eq_points_targets
    return gdf

def small_df(df, s=1000):
    df_small = df.sample(s, random_state=42)
    df_small = df_small.reset_index(drop=True)
    return df_small

def explore_gdf(gdf, m, color='green', radius=1, name=None):
    if len(gdf) > 0:
        gdf.explore(m=m,  marker_kwds={'radius': radius, 'color': color, 'fill': True, 'fill_color': color}, name=name)

def get_locations_and_map(data, country):
    gdf = data[data['Country'] == country]
    gdf_centroid = utils.gdf_center(gdf)
    gdf_north_west = get_north_west_gdf(gdf)
    gdf_south_east = get_south_east_gdf(gdf)
    gdf_north_east = get_north_east_gdf(gdf)
    gdf_south_west = get_south_west_gdf(gdf)

    m = folium.Map(location=[gdf_centroid.y , gdf_centroid.x], zoom_start=10)
    m = utils.add_esri_satellite_layer(m)
    m = utils.gdf_square_map(gdf, m=m)
    m = utils.gdf_square_map(gdf_north_west, m=m)
    m = utils.gdf_square_map(gdf_south_east, m=m)
    m = utils.gdf_square_map(gdf_north_east, m=m)
    m = utils.gdf_square_map(gdf_south_west, m=m)

    explore_gdf(gdf_north_west, m=m, color='black', radius=5, name='north_west')
    explore_gdf(gdf_south_east, m=m, color='white', radius=5, name='south_east')
    explore_gdf(gdf_north_east, m=m, color='red', radius=5, name='north_east')
    explore_gdf(gdf_south_west, m=m, color='blue', radius=5, name='south_west')
    explore_gdf(gdf, m=m, color='green', radius=1, name='all')
    folium.LayerControl().add_to(m)
    return gdf, m, gdf_north_west, gdf_south_east, gdf_north_east, gdf_south_west

def filter_date_by_month(dates, month):
    return [date for date in dates if date.split('-')[1] == month]

def filter_date_by_year(dates, year):
    return [date for date in dates if date.split('-')[0] == year]

def process_paths_df(paths_df, evalscript):
    current_image_paths_df = paths_df[paths_df['evalscript'] == evalscript]
    current_image_paths_df = current_image_paths_df.reset_index(drop=True)
    src_gdfs_evalscript = []
    if evalscript == 'ALL':
        bands = bands_all
    elif evalscript == 'FCOVER':
        bands = bands_fcover
    current_image_paths_df = current_image_paths_df.reset_index(drop=True)
    for i in tqdm(range(len(current_image_paths_df))):
        current_date = current_image_paths_df.iloc[i]['date']
        src_gdf = get_gdf_from_row(current_image_paths_df, i, bands_names=bands)
        new_names_dict = {}
        for col in src_gdf.columns:
            if col in bands:
                new_names_dict[col] = f'{col}_{current_date}'
            else:
                new_names_dict[col] = col
        src_gdf = src_gdf.rename(columns=new_names_dict)
        src_gdfs_evalscript.append(src_gdf)
    return src_gdfs_evalscript

def merege_src_gdf_columns_by_date(src_gdfs):
    src_gdfs_data_clos = []
    for src_gdf in src_gdfs:
        src_gdf = src_gdf.drop(columns=['Lat', 'Lon', 'geometry'])
        src_gdfs_data_clos.append(src_gdf)
    src_gdfs = pd.concat(src_gdfs_data_clos, axis=1)
    src_gdfs = src_gdfs.reset_index(drop=True)
    return src_gdfs

def get_processed_src_gdf(paths_df):
    src_gdfs_ALL_evalscript = process_paths_df(paths_df, 'ALL')
    src_gdfs_FCOVER_evalscript = process_paths_df(paths_df, 'FCOVER')
    lat_col = src_gdfs_ALL_evalscript[0]['Lat']
    lon_col = src_gdfs_ALL_evalscript[0]['Lon']
    geom_col = src_gdfs_ALL_evalscript[0]['geometry']
    src_gdfs_ALL_evalscript_mereged = merege_src_gdf_columns_by_date(src_gdfs_ALL_evalscript)
    src_gdfs_FCOVER_evalscript_mereged = merege_src_gdf_columns_by_date(src_gdfs_FCOVER_evalscript)
    src_gdf= pd.concat([src_gdfs_ALL_evalscript_mereged, src_gdfs_FCOVER_evalscript_mereged], axis=1)
    src_gdf['Location'] = paths_df['location'].values[0]
    src_gdf['Lat'] = lat_col
    src_gdf['Lon'] = lon_col
    src_gdf['geometry'] = geom_col
    src_gdf = gpd.GeoDataFrame(src_gdf, geometry='geometry')
    src_gdf.crs = 'EPSG:4326'
    src_gdf = src_gdf.reset_index(drop=True)
    return src_gdf

def get_processed_src_gdf_fcover(paths_df):
    src_gdf = process_paths_df(paths_df, 'FCOVER')
    src_gdf = src_gdf[0]
    src_gdf['Location'] = paths_df['location'].values[0]
    src_gdf = gpd.GeoDataFrame(src_gdf, geometry='geometry')
    src_gdf.crs = 'EPSG:4326'
    src_gdf = src_gdf.reset_index(drop=True)
    return src_gdf

def get_masking_dict_for_src_from_target(src_geom, target_gdf, dict_path):
    if os.path.exists(dict_path):
        calculated_dict = joblib.load(dict_path)
    else:
        target_geom = target_gdf.geometry.values
        target_gdf_cols = target_gdf.columns
        if 'Target' in target_gdf_cols:
            target_labels = target_gdf.Target.values
        else:
            target_labels = [-1 for i in range(len(target_geom))]
        eq_points_indicies = []
        eq_points_targets = []
        j = 0
        for target_point in tqdm(target_geom):
            t = target_labels[j]
            j+=1
            i = 0
            target_point_matched = False
            for src_point in src_geom:
                if src_point.equals_exact(target_point, 1e-4):
                    eq_points_indicies.append(i)
                    eq_points_targets.append(t)
                    target_point_matched = True
                    break
                i+=1
            if not target_point_matched:
                print(f'No match for target_point: {target_point} with t: {t} and j: {j}')
                eq_points_indicies.append(-1)
                eq_points_targets.append(t)
        calculated_dict = {
            'eq_points_indicies': eq_points_indicies,
            'eq_points_targets': eq_points_targets,
        }
        joblib.dump(calculated_dict, dict_path)
    return calculated_dict

def decrease_gdf_height_by_removing_max_and_min_lats(gdf, n=1):
    if n > 0:
        gdf = gdf.reset_index(drop=True)
        lats = gdf['Lat'].values
        sorted_lats_indicies = np.argsort(lats)
        n_max = sorted_lats_indicies[-n:]
        n_min = sorted_lats_indicies[:n]
        print(f'Max indicies: {n_max}, Max lats: {lats[n_max]}')
        print(f'Min indicies: {n_min}, Min lats: {lats[n_min]}')
        #drop rows with max and min lats indicies
        gdf = gdf.drop(n_max)
        gdf = gdf.drop(n_min)
        gdf = gdf.reset_index(drop=True)
    return gdf

def decrease_gdf_width_by_removing_max_and_min_lons(gdf, m=1):
    if m > 0:
        gdf = gdf.reset_index(drop=True)
        lons = gdf['Lon'].values
        sorted_lons_indicies = np.argsort(lons)
        m_max = sorted_lons_indicies[-m:]
        m_min = sorted_lons_indicies[:m]
        print(f'Max indicies: {m_max}, Max lons: {lons[m_max]}')
        print(f'Min indicies: {m_min}, Min lons: {lons[m_min]}')
        #drop rows with max and min lons indicies
        gdf = gdf.drop(m_max)
        gdf = gdf.drop(m_min)
        gdf = gdf.reset_index(drop=True)
    return gdf

def decrease_gdf_height_and_width(gdf, n, m):
    gdf = decrease_gdf_height_by_removing_max_and_min_lats(gdf, n)
    gdf = decrease_gdf_width_by_removing_max_and_min_lons(gdf, m)
    return gdf

def reszie_quarter_gdfs(country_quarter_split_gdf, country_quarter_split_gdf_name):
    if country_quarter_split_gdf_name == 'Iran_North_West_test':
        n = 1
        m = 0
    elif country_quarter_split_gdf_name == 'Iran_South_East_test':
        n = 8
        m = 0
    elif country_quarter_split_gdf_name == 'Iran_North_East_test':
        n = 5
        m = 0
    elif country_quarter_split_gdf_name == 'Iran_South_West_test':
        n = 3
        m = 0
    elif country_quarter_split_gdf_name == 'Sudan_North_West_test':
        n = 4
        m = 5
    elif country_quarter_split_gdf_name == 'Sudan_South_East_test':
        n = 2
        m = 1
    elif country_quarter_split_gdf_name == 'Sudan_North_East_test':
        n = 0
        m = 1
    elif country_quarter_split_gdf_name == 'Sudan_South_West_test':
        n = 4
        m = 1
    elif country_quarter_split_gdf_name == 'Sudan_North_West_train':
        n = 0
        m = 1
    elif country_quarter_split_gdf_name == 'Sudan_South_East_train':
        n = 0
        m = 1
    elif country_quarter_split_gdf_name == 'Sudan_North_East_train':
        n = 0
        m = 3
    elif country_quarter_split_gdf_name == 'Sudan_South_West_train':
        n = 1
        m = 0
    else:
        n = 0
        m = 0
    country_quarter_split_gdf = decrease_gdf_height_and_width(country_quarter_split_gdf, n, m)
    return country_quarter_split_gdf
        
def creat_masking_dict(target_gdf, country, direction, train_or_test, target_location, part2=False):
    processed_gdf_path = f'{country}_{direction}_{train_or_test}_processed_gdf.joblib'
    if os.path.exists(processed_gdf_path):
        print(f'processed_gdf_path: {processed_gdf_path} exists')
        return None
    dir = './data/masking_dicts'
    if not os.path.exists(dir):
        os.mkdir(dir)
    dict_path = f'./data/masking_dicts/{country}_{direction}_{train_or_test}_calculated_dict.joblib'
    images_paths_df = utils.get_available_data_dataframe()
    images_paths_df = images_paths_df[images_paths_df['location'] == target_location]
    images_paths_df = images_paths_df.reset_index(drop=True)
    if len(images_paths_df) == 0:
        print(f'No images for {target_location}')
        return None
    evalscripts = images_paths_df['evalscript'].values
    indices = [i for i in range(len(evalscripts)) if evalscripts[i] in ['FCOVER', 'ALL']]
    images_paths_df = images_paths_df.iloc[indices]
    images_paths_df = images_paths_df.reset_index(drop=True)
    all_dates = images_paths_df['date'].unique()
    dates_2019 = filter_date_by_year(all_dates, '2019')
    dates_2020 = filter_date_by_year(all_dates, '2020')
    monhts_2019 = ['07', '10', '12']
    monhts_2020 = ['02', '04', '06']
    dates = []
    for month in monhts_2019:
        dates += filter_date_by_month(dates_2019, month)
    for month in monhts_2020:
        dates += filter_date_by_month(dates_2020, month)
    if dates is not None:
        images_paths_df = images_paths_df[images_paths_df['date'].isin(dates)]
        images_paths_df = images_paths_df.reset_index(drop=True)
    if part2:
        print(f'Getting masking_dict for {target_location} part 2: {len(images_paths_df)}')
        print('Getting processed_src_gdf')
        processed_src_gdf = get_processed_src_gdf(images_paths_df)
        src_geom = processed_src_gdf.geometry.values
        masking_dict = get_masking_dict_for_src_from_target(src_geom, target_gdf, dict_path)
        processed_masked_src_gdf = get_processed_masked_src_gdf(processed_src_gdf, masking_dict)
        joblib.dump(processed_masked_src_gdf, processed_gdf_path)
        print(f'processed_masked_src_gdf: {len(processed_masked_src_gdf)} saved to {processed_gdf_path}')
    else:
        processed_src_gdf = get_processed_src_gdf_fcover(images_paths_df)
        src_geom = processed_src_gdf.geometry.values
        masking_dict = get_masking_dict_for_src_from_target(src_geom, target_gdf, dict_path)
        return masking_dict

def pre_prep_masking_jobs():
    train_or_test_values = ['train', 'test']
    country_values = ['Sudan', 'Iran']
    direction_values = ['north_west', 'south_east', 'north_east', 'south_west']
    for country in country_values:
        print(f'country: {country}')
        for train_or_test in train_or_test_values:
            print(f'train_or_test: {train_or_test}')
            if train_or_test == 'train':
                data = utils.read_train_data()
            elif train_or_test == 'test':
                data = utils.read_test_data()
            gdf, m, gdf_north_west, gdf_south_east, gdf_north_east, gdf_south_west = get_locations_and_map(data, country)
            country_north_west = f'{country}_North_West_{train_or_test}'
            country_south_east = f'{country}_South_East_{train_or_test}'
            country_north_east = f'{country}_North_East_{train_or_test}'
            country_south_west = f'{country}_South_West_{train_or_test}'
            gdf_north_west = reszie_quarter_gdfs(gdf_north_west, country_north_west)
            gdf_south_east = reszie_quarter_gdfs(gdf_south_east, country_south_east)
            gdf_north_east = reszie_quarter_gdfs(gdf_north_east, country_north_east)
            gdf_south_west = reszie_quarter_gdfs(gdf_south_west, country_south_west)
            for direction in direction_values:
                print(f'direction: {direction}')
                masking_dict_job_args_path = f'./data/masking_jobs/{country}_{direction}_{train_or_test}_masking_dict_args.joblib'
                if os.path.exists(masking_dict_job_args_path):
                    continue
                if direction == 'north_west':
                    target_gdf = gdf_north_west.copy()
                    target_location = country_north_west
                elif direction == 'south_east':
                    target_gdf = gdf_south_east.copy()
                    target_location = country_south_east
                elif direction == 'north_east':
                    target_gdf = gdf_north_east.copy()
                    target_location = country_north_east
                elif direction == 'south_west':
                    target_gdf = gdf_south_west.copy()
                    target_location = country_south_west
                masking_dict_args = {
                    'target_gdf': target_gdf,
                    'country': country,
                    'direction': direction,
                    'train_or_test': train_or_test,
                    'target_location': target_location
                }
                joblib.dump(masking_dict_args, masking_dict_job_args_path)

def main_parallel_data_processor(masking_dict_job_args_path):
    masking_dict_args = joblib.load(masking_dict_job_args_path)
    target_gdf = masking_dict_args['target_gdf']
    country = masking_dict_args['country']
    direction = masking_dict_args['direction']
    train_or_test = masking_dict_args['train_or_test']
    target_location = masking_dict_args['target_location']
    masking_dict = creat_masking_dict(target_gdf, country, direction, train_or_test, target_location)
    # os.remove(masking_dict_job_args_path)
    return masking_dict

def main_parallel_data_processor_part_2(masking_dict_job_args_path):
    masking_dict_args = joblib.load(masking_dict_job_args_path)
    target_gdf = masking_dict_args['target_gdf']
    country = masking_dict_args['country']
    direction = masking_dict_args['direction']
    train_or_test = masking_dict_args['train_or_test']
    target_location = masking_dict_args['target_location']
    masking_dict = creat_masking_dict(target_gdf, country, direction, train_or_test, target_location, part2=True)

    # os.remove(masking_dict_job_args_path)
    return masking_dict

def main():
    # masking_dict_job_args_dir = './data/masking_jobs'
    # masking_dict_job_args_files = os.listdir(masking_dict_job_args_dir)
    # masking_dict_job_args_paths = [os.path.join(masking_dict_job_args_dir, masking_dict_job_args_file) for masking_dict_job_args_file in masking_dict_job_args_files]
    # for i in range(len(masking_dict_job_args_paths)):
    #     print(f'{i}: {masking_dict_job_args_paths[i]}')
    # print(f'Number of masking_dict_job_args_paths: {len(masking_dict_job_args_paths)}')

    # with Pool(processes=multiprocessing.cpu_count()) as pool:
    #     pool.map(main_parallel_data_processor, masking_dict_job_args_paths)
    # print('Done!')

    ###Part 2
    masking_dict_job_args_dir = './data/masking_jobs'
    masking_dict_job_args_files = os.listdir(masking_dict_job_args_dir)
    masking_dict_job_args_paths = [os.path.join(masking_dict_job_args_dir, masking_dict_job_args_file) for masking_dict_job_args_file in masking_dict_job_args_files]
    print(f'Number of masking_dict_job_args_paths: {len(masking_dict_job_args_paths)}')
    # with Pool(processes=multiprocessing.cpu_count()) as pool:
    #     pool.map(main_parallel_data_processor_part_2, masking_dict_job_args_paths)
    for i in tqdm(range(len(masking_dict_job_args_paths))):
        main_parallel_data_processor_part_2(masking_dict_job_args_paths[i])
    print('Done!')

if __name__ == '__main__':
    main()