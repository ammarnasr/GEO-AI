import sys
import mask
import utils
import train
import joblib
import pandas as pd
import geopandas as gpd

def get_afghanistan_train_data_at_date(date):
    image_paths_df  = utils.get_available_data_dataframe()
    evalscript_filter = ['ALL', 'FCOVER']
    image_paths_df = image_paths_df[image_paths_df['evalscript'].isin(evalscript_filter)]
    location_filter = ['Afghanistan_North_West_train', 'Afghanistan_South_East_train']
    image_paths_df = image_paths_df[image_paths_df['location'].isin(location_filter)]
    image_paths_df = image_paths_df.reset_index(drop=True)
    date_filter = [date]
    image_paths_df = image_paths_df[image_paths_df['date'].isin(date_filter)]
    image_paths_df = image_paths_df.reset_index(drop=True)
    bands_all = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08','B8A', 'B09', 'B10', 'B11', 'B12']
    bands_fcover = ['FCOVER']
    north_west_train_all_index = 0
    north_west_train_fcover_index = 1
    south_east_train_all_index = 2
    south_east_train_fcover_index = 3
    gdf_north_west_train_all = mask.get_gdf_from_row(image_paths_df, north_west_train_all_index, bands_all)
    gdf_north_west_train_fcover = mask.get_gdf_from_row(image_paths_df, north_west_train_fcover_index, bands_fcover)
    gdf_south_east_train_all = mask.get_gdf_from_row(image_paths_df, south_east_train_all_index, bands_all)
    gdf_south_east_train_fcover = mask.get_gdf_from_row(image_paths_df, south_east_train_fcover_index, bands_fcover)
    gdf_north_west_train_all['FCOVER'] = gdf_north_west_train_fcover['FCOVER']
    gdf_south_east_train_all['FCOVER'] = gdf_south_east_train_fcover['FCOVER']
    return [gdf_north_west_train_all, gdf_south_east_train_all]


def get_afghanistan_test_data_at_date(date):
    image_paths_df  = utils.get_available_data_dataframe()
    evalscript_filter = ['ALL', 'FCOVER']
    image_paths_df = image_paths_df[image_paths_df['evalscript'].isin(evalscript_filter)]
    location_filter = ['Afghanistan_North_West_test', 'Afghanistan_South_East_test']
    image_paths_df = image_paths_df[image_paths_df['location'].isin(location_filter)]
    image_paths_df = image_paths_df.reset_index(drop=True)
    date_filter = [date]
    image_paths_df = image_paths_df[image_paths_df['date'].isin(date_filter)]
    image_paths_df = image_paths_df.reset_index(drop=True)
    bands_all = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08','B8A', 'B09', 'B10', 'B11', 'B12']
    bands_fcover = ['FCOVER']
    north_west_test_all_index = 0
    north_west_test_fcover_index = 1
    south_east_test_all_index = 2
    south_east_test_fcover_index = 3
    gdf_north_west_test_all    = mask.get_gdf_from_row(image_paths_df, north_west_test_all_index, bands_all)
    gdf_north_west_test_fcover = mask.get_gdf_from_row(image_paths_df, north_west_test_fcover_index, bands_fcover)
    gdf_south_east_test_all    = mask.get_gdf_from_row(image_paths_df, south_east_test_all_index, bands_all)
    gdf_south_east_test_fcover = mask.get_gdf_from_row(image_paths_df, south_east_test_fcover_index, bands_fcover)
    gdf_north_west_test_all['FCOVER'] = gdf_north_west_test_fcover['FCOVER']
    gdf_south_east_test_all['FCOVER'] = gdf_south_east_test_fcover['FCOVER']
    return [gdf_north_west_test_all, gdf_south_east_test_all]


def process_splits_afghanistan_train(splits):
    afghanistan_train_gdf_north_west = splits[0]
    afghanistan_train_gdf_south_east = splits[1]
    d_north_west_path = 'Afghanistan_north_west_train_calculated_dict.joblib'
    d_south_east_path = 'Afghanistan_south_east_train_calculated_dict.joblib'
    d_north_west = joblib.load(d_north_west_path)
    d_south_east = joblib.load(d_south_east_path)
    eq_points_inds_north_west = d_north_west['eq_points_indicies']
    eq_points_inds_south_east = d_south_east['eq_points_indicies']
    eq_points_targets_north_west = d_north_west['eq_points_targets']
    eq_points_targets_south_east = d_south_east['eq_points_targets']
    afghanistan_train_gdf_north_west_eq = afghanistan_train_gdf_north_west.iloc[eq_points_inds_north_west]
    afghanistan_train_gdf_south_east_eq = afghanistan_train_gdf_south_east.iloc[eq_points_inds_south_east]
    afghanistan_train_gdf_north_west_eq['Target'] = eq_points_targets_north_west
    afghanistan_train_gdf_south_east_eq['Target'] = eq_points_targets_south_east
    afghanistan_train_gdf_north_west_eq = afghanistan_train_gdf_north_west_eq.reset_index(drop=True)
    afghanistan_train_gdf_south_east_eq = afghanistan_train_gdf_south_east_eq.reset_index(drop=True)
    afghanistan_train_gdf = pd.concat([afghanistan_train_gdf_north_west_eq, afghanistan_train_gdf_south_east_eq], ignore_index=True)
    afghanistan_train_gdf = gpd.GeoDataFrame(afghanistan_train_gdf, geometry='geometry')
    afghanistan_train_gdf.crs = 'EPSG:4326'
    afghanistan_train_gdf = afghanistan_train_gdf.reset_index(drop=True)
    return afghanistan_train_gdf


def process_splits_afghanistan_test(splits):
    afghanistan_test_gdf_north_west = splits[0]
    afghanistan_test_gdf_south_east = splits[1]
    d_north_west_path = 'Afghanistan_north_west_test_calculated_dict.joblib'
    d_south_east_path = 'Afghanistan_south_east_test_calculated_dict.joblib'
    d_north_west = joblib.load(d_north_west_path)
    d_south_east = joblib.load(d_south_east_path)
    eq_points_inds_north_west = d_north_west['eq_points_indicies']
    eq_points_inds_south_east = d_south_east['eq_points_indicies']
    eq_points_targets_north_west = d_north_west['eq_points_targets']
    eq_points_targets_south_east = d_south_east['eq_points_targets']
    afghanistan_test_gdf_north_west_eq = afghanistan_test_gdf_north_west.iloc[eq_points_inds_north_west]
    afghanistan_test_gdf_south_east_eq = afghanistan_test_gdf_south_east.iloc[eq_points_inds_south_east]
    afghanistan_test_gdf_north_west_eq['Target'] = eq_points_targets_north_west
    afghanistan_test_gdf_south_east_eq['Target'] = eq_points_targets_south_east
    afghanistan_test_gdf_north_west_eq = afghanistan_test_gdf_north_west_eq.reset_index(drop=True)
    afghanistan_test_gdf_south_east_eq = afghanistan_test_gdf_south_east_eq.reset_index(drop=True)
    afghanistan_test_gdf = pd.concat([afghanistan_test_gdf_north_west_eq, afghanistan_test_gdf_south_east_eq], ignore_index=True)
    afghanistan_test_gdf = gpd.GeoDataFrame(afghanistan_test_gdf, geometry='geometry')
    afghanistan_test_gdf.crs = 'EPSG:4326'
    afghanistan_test_gdf = afghanistan_test_gdf.reset_index(drop=True)
    return afghanistan_test_gdf


def train_afghanistan_model(classifer_key,classifer_key_name='low_cloud', random_cv_iteraions=-1):
    joblib_path_train = './data/joblibs/afghanistan_train_all_dates.joblib'
    joblib_path_test = './data/joblibs/afghanistan_test_all_dates.joblib'
    train_gdf_old = joblib.load(joblib_path_train)
    test_gdf_old = joblib.load(joblib_path_test)
    low_cloud_dates = ['2022-01-13', '2022-02-12', '2022-03-04', '2022-04-08' ,'2022-05-23']
    columns_with_dates_train = [c for c in train_gdf_old.columns if '2022' in c]
    columns_with_dates_test = [c for c in test_gdf_old.columns if '2022' in c]
    columns_without_dates_train = [c for c in train_gdf_old.columns if '2022' not in c]
    columns_without_dates_test = [c for c in test_gdf_old.columns if '2022' not in c]
    low_cloud_columns_train = [c for c in columns_with_dates_train if c.split('_')[-1] in low_cloud_dates]
    low_cloud_columns_test = [c for c in columns_with_dates_test if c.split('_')[-1] in low_cloud_dates]
    train_gdf_low_cloud = train_gdf_old[low_cloud_columns_train]
    test_gdf_low_cloud = test_gdf_old[low_cloud_columns_test]
    for c in columns_without_dates_train:
        train_gdf_low_cloud[c] = train_gdf_old[c]
    for c in columns_without_dates_test:
        test_gdf_low_cloud[c] = test_gdf_old[c]
    country = 'Afghanistan'
    _, _, train_original, test_original = train.get_processed_data_for_country(country)
    train_gdf_low_cloud_with_id, test_gdf_low_cloud_with_id = train.train_and_test_gdfs_matched_to_original_train_test(
        train_gdf_low_cloud, test_gdf_low_cloud, train_original, test_original, country, use_cached=False)
    classifiers_dict = train.get_calssifiers_dict()
    drop_cols = ['Location', 'geometry', 'Lat', 'Lon', 'Target', 'ID']
    X_train = train_gdf_low_cloud_with_id.drop(columns=drop_cols)
    y_train = train_gdf_low_cloud_with_id['Target']
    args_dict = classifiers_dict[classifer_key]['args_dict']
    param_grid = classifiers_dict[classifer_key]['param_grid']
    classifier_name = classifiers_dict[classifer_key]['classifier_name']
    cv = classifiers_dict[classifer_key]['cv']
    best_params, best_score, best_model = train.grid_search(X_train, y_train, args_dict, param_grid, cv, classifier_name, verbose=4, random_cv_iterations=random_cv_iteraions)
    classifiers_dict[classifer_key]['best']={}
    classifiers_dict[classifer_key]['best']['param']=best_params
    classifiers_dict[classifer_key]['best']['acc']=best_score
    classifiers_dict[classifer_key]['best']['model']=best_model
    print(f'best_params: {best_params}')
    print(f'best_score: {best_score}')
    print(f'best_model: {best_model}')
    trained_classifier_dict = classifiers_dict[classifer_key]
    best_model = trained_classifier_dict['best']['model']
    test_with_ids_preds = train.save_best_model_and_create_submession(
        train_gdf_low_cloud_with_id, test_gdf_low_cloud_with_id, country,best_model, classifer_key_name, do_save=True)
    return test_with_ids_preds


if __name__ == '__main__':
    switch_variable = len(sys.argv)
    if switch_variable == 1:
        print('No arguments given')
    elif switch_variable == 2:
        classifer_key = sys.argv[1]
        train_afghanistan_model(classifer_key)
    elif switch_variable == 3:
        classifer_key = sys.argv[1]
        classifer_key_name = sys.argv[2]
        train_afghanistan_model(classifer_key, classifer_key_name)
    elif switch_variable == 4:
        classifer_key = sys.argv[1]
        classifer_key_name = sys.argv[2]
        random_cv_iteraions = int(sys.argv[3])
        train_afghanistan_model(classifer_key, classifer_key_name, random_cv_iteraions)