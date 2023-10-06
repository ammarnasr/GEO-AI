import os
import sys
import utils
import utils
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import geopandas as gpd
from tqdm.auto import tqdm
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier


def train_with_pipline(X_train, y_train):
    clf2 = RandomForestClassifier()
    clf6 = AdaBoostClassifier()
    clf7 = GradientBoostingClassifier()
    eclf = VotingClassifier(estimators=[('rf', clf2), ('xgb', clf7), ('ab', clf6)], voting='soft', n_jobs=-1)
    calssifiers_dict = get_calssifiers_dict()
    params = {
        'rf__n_estimators': calssifiers_dict['RF']['param_grid']['n_estimators'],
        'rf__max_depth': calssifiers_dict['RF']['param_grid']['max_depth'],
        'xgb__n_estimators': calssifiers_dict['XGB']['param_grid']['n_estimators'],
        'xgb__max_depth': calssifiers_dict['XGB']['param_grid']['max_depth'],
        'ab__n_estimators': calssifiers_dict['AB']['param_grid']['n_estimators'],
        'ab__learning_rate': calssifiers_dict['AB']['param_grid']['learning_rate'],
        'weights': [[1, 1, 1], [5,1,1]],
        'voting': ['soft'],
        'n_jobs': [-1]

        }
    start_time = datetime.now()
    # Define your GridSearchCV object
    # grid_search = GridSearchCV(eclf, params, cv=5, scoring='accuracy', n_jobs=-1, verbose=3)
    grid_search = RandomizedSearchCV(eclf, params, n_iter=30, cv=5, scoring='accuracy', n_jobs=-1, verbose=3)
    # Fit your GridSearchCV object to your training data
    grid_search.fit(X_train, y_train)
    # Get the best estimator and print out its accuracy score
    endtime = datetime.now()
    print('=================================================')
    print(f'Time elapsed: {endtime - start_time}')
    print('=================================================')
    best_score = grid_search.best_score_
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_
    return best_params, best_score, best_model

def add_ndvi_cols(df):
    #ndvi = (B08 - B04) / (B08 + B04)
    cols = df.columns
    B04_cols = [col for col in cols if 'B04' in col]
    B08_cols = [col for col in cols if 'B08' in col]
    date_suffixes = [col.split('_')[-1] for col in B04_cols]
    ndvi_cols = [f'ndvi_{date_suffix}' for date_suffix in date_suffixes]
    for ndvi_col, B04_col, B08_col in zip(ndvi_cols, B04_cols, B08_cols):
        df[ndvi_col] = (df[B08_col] - df[B04_col]) / (df[B08_col] + df[B04_col])
    return df

def add_arvi(df):
    #y = 0.106
    #arvi = (B8A - B04 - y * (B04 - B02)) / (B8A + B04 - y * (B04 - B02))
    cols = df.columns
    y = 0.106
    B02_cols = [col for col in cols if 'B02' in col]
    B04_cols = [col for col in cols if 'B04' in col]
    B08A_cols = [col for col in cols if 'B08A' in col]
    date_suffixes = [col.split('_')[-1] for col in B04_cols]
    arvi_cols = [f'arvi_{date_suffix}' for date_suffix in date_suffixes]
    for arvi_col, B02_col, B04_col, B08A_col in zip(arvi_cols, B02_cols, B04_cols, B08A_cols):
        df[arvi_col] = (df[B08A_col] - df[B04_col] - y * (df[B04_col] - df[B02_col])) / (df[B08A_col] + df[B04_col] - y * (df[B04_col] - df[B02_col]))
    return df

def normalize_df_by_mean_and_std(df):
    #normalize df by mean and std
    cols = df.columns
    cols = [col for col in cols if '-' in col and '_' in col]
    for col in cols:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = (df[col] - mean) / std
    return df

def drop_cols_for_clf(gdf, drop_cols=None):
    if drop_cols is None:
        drop_cols = ['Location', 'geometry', 'Lat', 'Lon', 'Target', 'ID']
    gdf = gdf.drop(columns=drop_cols)
    return gdf

def get_calssifiers_dict():
    classifiers_dict = {
        'RF':{
            'classifier_name': 'RandomForestClassifier',
            'args_dict': {'n_estimators': 300, 'max_depth': 30, 'n_jobs': -1, 'verbose': 0},
            'param_grid': {
                'n_estimators': [70,80, 90],
                'max_depth': [14,15, 16]
                },
            'cv': StratifiedKFold(n_splits=5, shuffle=True)},
        'XGB':{
            'classifier_name': 'GradientBoostingClassifier',
            'args_dict': {'n_estimators': 300, 'max_depth': 30, 'verbose': 0},

            'param_grid': {
                'n_estimators': [80, 90, 100],
                'max_depth': [9,8,10]
                },
            'cv': StratifiedKFold(n_splits=5, shuffle=True)},
        'AB':{
            'classifier_name': 'AdaBoostClassifier',
            'args_dict': {'n_estimators': 300, 'random_state': 0},
            'param_grid': {
                'n_estimators': [800, 900, 1000],
                'learning_rate': [1.0, 1.1, 1.2],
                },
            'cv': StratifiedKFold(n_splits=5, shuffle=True, random_state=0)},
        'NB':{
            'classifier_name': 'GaussianNB',
            'args_dict': {'var_smoothing': 1e-09},
            'param_grid': {
                'var_smoothing': [1e-09, 1e-08, 1e-07],
                },
            'cv': StratifiedKFold(n_splits=5, shuffle=True)},
        'MLP':{
            'classifier_name': 'MLPClassifier',
            'args_dict': {'hidden_layer_sizes': (100, 100, 100), 'max_iter': 1000},
            'param_grid': {
                'hidden_layer_sizes': [(500,500), (500, 500, 500), (100, 100, 100), (1000, 1000)],
                # 'max_iter': [100, 500, 1000],
                # 'activation': ['relu', 'tanh', 'logistic'],
                # 'solver': ['adam', 'sgd', 'lbfgs'],
                # 'learning_rate': ['constant', 'invscaling', 'adaptive'],
                'learning_rate_init': [0.0005, 0.001, 0.005],
                },
            'cv': StratifiedKFold(n_splits=5, shuffle=True)},
        'STACK':{
            'classifier_name': 'StackingClassifier',
            'args_dict': {'estimators': [
                ('rf', RandomForestClassifier(n_estimators=80, max_depth=15, n_jobs=-1, verbose=0)),
                ('xgb', GradientBoostingClassifier(n_estimators=90, max_depth=8, verbose=0)),
                ('ab', AdaBoostClassifier(n_estimators=900, learning_rate=1.1, random_state=0)),
                ('mlp', MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=1000))
                ],
                'final_estimator': LogisticRegression()},
            'param_grid': {
                'final_estimator__C': [0.1, 0.5, 1.0, 1.5, 2.0],
                'final_estimator__solver': ['lbfgs', 'liblinear'],
                'stack_method': ['auto', 'predict_proba'],
                'passthrough': [True, False],
                'n_jobs': [-1],
                'rf__n_estimators': [70,80, 90],
                'rf__max_depth': [14,15, 16],
                'xgb__n_estimators': [80, 90, 100],
                'xgb__max_depth': [9,8,10],
                'ab__n_estimators': [800, 900, 1000],
                'ab__learning_rate': [1.0, 1.1, 1.2],
                'mlp__hidden_layer_sizes': [(500,500), (500, 500, 500), (100, 100, 100), (1000, 1000)],
                'mlp__learning_rate_init': [0.0005, 0.001, 0.005],
                },
            'cv': StratifiedKFold(n_splits=5, shuffle=True)},
        'VOTE':{
            'classifier_name': 'VotingClassifier',
            'args_dict': {
                'estimators': [
                    ('rf', RandomForestClassifier(n_estimators=80, max_depth=15, n_jobs=-1, verbose=0)),
                    ('xgb', GradientBoostingClassifier(n_estimators=90, max_depth=8, verbose=0)),
                    ('ab', AdaBoostClassifier(n_estimators=900, learning_rate=1.1, random_state=0))],},
            'param_grid': {
                'weights': [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]],
                'voting': ['soft', 'hard'],
                'n_jobs': [-1]
                },
            'cv': StratifiedKFold(n_splits=5, shuffle=True)},
    }
    return classifiers_dict

def get_orginal_train_test(country):
    test_original = utils.read_test_data()
    test_original = test_original[test_original['Country'] == country]
    test_original = test_original.reset_index(drop=True)
    train_original = utils.read_train_data()
    train_original = train_original[train_original['Country'] == country]
    train_original = train_original.reset_index(drop=True)
    return train_original, test_original

def find_match(src_lat, src_lon, dst_lat_lon, round_to=3):
    src_lat = round(src_lat, round_to)
    src_lon = round(src_lon, round_to)
    matches = []
    match_indices = []
    i = 0
    for dst_lat, dst_lon in dst_lat_lon:
        dst_lat = round(dst_lat, round_to)
        dst_lon = round(dst_lon, round_to)
        if src_lat == dst_lat and src_lon == dst_lon:
            matches.append((dst_lat, dst_lon))
            match_indices.append(i)
        i += 1
    if len(matches) > 0:
        return True, matches, match_indices
    return False, matches, match_indices

def match_src_test_to_orginal_test(gdf_test, country):
    test = utils.read_test_data()
    test = test[test['Country'] == country]
    test = test.reset_index(drop=True)
    gdf_test_lat = gdf_test['Lat']
    gdf_test_lon = gdf_test['Lon']
    test_lat = test['Lat']
    test_lon = test['Lon']
    dst_lat_lon = [(lat, lon) for lat, lon in zip(gdf_test_lat, gdf_test_lon)]
    src_lat_lon = [(lat, lon) for lat, lon in zip(test_lat, test_lon)]
    all_match_indices = []
    i = 0
    for src_lat, src_lon in src_lat_lon:
        do_match, _, match_indices = find_match(src_lat, src_lon, dst_lat_lon, round_to=3)
        if do_match:
            all_match_indices.append(match_indices)
        else:
            # print(f'No match for: index {i} at round_to=3')
            do_match, _, match_indices = find_match(src_lat, src_lon, dst_lat_lon, round_to=2)
            if do_match:
                all_match_indices.append(match_indices)
            else:
                # print(f'No match for: index {i} at round_to=2')
                do_match, _, match_indices = find_match(src_lat, src_lon, dst_lat_lon, round_to=1)
                if do_match:
                    all_match_indices.append(match_indices)
                else:
                    print(f'No match for:  index {i} at round_to=1')
        i += 1
    final_match_indices = [m[0] for m in all_match_indices]
    gdf_test = gdf_test.iloc[final_match_indices]
    gdf_test = gdf_test.reset_index(drop=True)
    gdf_test['ID'] = test['ID']
    return gdf_test, test

def create_submession(preds, submession_name='sample_submission.csv'):
    test = pd.read_csv('Test.csv')
    test['target'] = preds
    sub_file = pd.DataFrame({'ID': test.ID, 'Target': preds})
    sub_file.to_csv(submession_name, index=False)
    print('Saved file: ' + submession_name)
    return sub_file

def get_all_processed_data_for_country_for_split_in_dir(country, split, data_dir):
    files_in_data_dir = os.listdir(data_dir)
    country_files = [file for file in files_in_data_dir if country in file]
    country_files_processed_gdf = [file for file in country_files if 'processed_gdf' in file]
    country_files_processed_gdf_train = [file for file in country_files_processed_gdf if 'train' in file]
    country_files_processed_gdf_test = [file for file in country_files_processed_gdf if 'test' in file]
    if split == 'train':
        gdfs = [joblib.load(file) for file in country_files_processed_gdf_train]
    elif split == 'test':
        gdfs = [joblib.load(file) for file in country_files_processed_gdf_test]
    else:
        gdfs = []
    gdf = pd.concat(gdfs, axis=0)
    gdf = gdf.reset_index(drop=True)
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry')
    gdf.crs = 'EPSG:4326'
    gdf = gdf.to_crs('EPSG:4326')
    gdf = gdf.reset_index(drop=True)
    return gdf

def create_submession(test, submession_name='sample_submission.csv'):
    test_final = utils.read_test_data()
    test_ids = test['ID'].values
    test_final_ids = test_final['ID'].values
    test_targets = test['Target'].values
    test_final_targets = []
    for test_final_id in test_final_ids:
        if test_final_id in test_ids:
            id_index = test_ids.tolist().index(test_final_id)
            test_final_targets.append(test_targets[id_index])
        else:
            test_final_targets.append(-1)

    sub_file = pd.DataFrame({'ID': test_final_ids, 'Target': test_final_targets})
    sub_file.to_csv(submession_name, index=False)
    print('Saved file: ' + submession_name)
    return sub_file

def get_porcessed_data_dict():
    data_dir = './'
    countries = ['Sudan', 'Iran', 'Afghanistan']
    splits = ['train', 'test']
    porcessed_data_dict = {}
    for country in countries:
        for split in splits:
            gdf = get_all_processed_data_for_country_for_split_in_dir(country, split, data_dir)
            if country in porcessed_data_dict:
                porcessed_data_dict[country][split] = gdf
            else:
                porcessed_data_dict[country] = {split: gdf}
            print(f'Country {country}, split {split}: {gdf.shape}')    
    return porcessed_data_dict

def get_final_train_test_gdfs(porcessed_data_dict, country):
    gdf_train = porcessed_data_dict[country]['train']
    gdf_test = porcessed_data_dict[country]['test']
    gdf_test, test = match_src_test_to_orginal_test(gdf_test, country)
    na_cols = gdf_train.isna().sum()
    dropping_cols = []
    for col in na_cols.index:
        if na_cols[col] > 0:
            dropping_cols.append(col)
    gdf_train = gdf_train.drop(columns=dropping_cols)
    gdf_test = gdf_test.drop(columns=dropping_cols)
    na_cols = gdf_train.isna().sum().sum()
    na_cols = gdf_test.isna().sum().sum()
    print(f'gdf_train: {gdf_train.shape}, gdf_test: {gdf_test.shape}, test_with_ids: {test.shape}')
    return gdf_train, gdf_test, test

def get_train_test_splits_X_Y(gdf, test_size, drop_cols=None, random_state=None):
    if drop_cols is None:
        drop_cols = ['Location', 'geometry', 'Lat', 'Lon', 'Target', 'ID']
    X = gdf.drop(columns=drop_cols)
    y = gdf['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def train_classifier(X_train, y_train, args_dict, classifier_name='RandomForestClassifier'):
    if classifier_name == 'RandomForestClassifier':
        clf = RandomForestClassifier(**args_dict)
    clf.fit(X_train, y_train)
    return clf

def grid_search(X_train, y_train, args_dict, param_grid, cv,classifier_name='RandomForestClassifier', verbose=4, random_cv_iterations=-1):
    if classifier_name == 'RandomForestClassifier':
        clf = RandomForestClassifier(**args_dict)
    elif classifier_name == 'AdaBoostClassifier':
        clf = AdaBoostClassifier(**args_dict)
    elif classifier_name == 'GradientBoostingClassifier':
        clf = GradientBoostingClassifier(**args_dict)
    elif classifier_name == 'GaussianNB':
        clf = GaussianNB(**args_dict)
    elif classifier_name == 'MLPClassifier':
        clf = MLPClassifier(**args_dict)
    elif classifier_name == 'StackingClassifier':
        clf = StackingClassifier(**args_dict)
    elif classifier_name == 'VotingClassifier':
        clf = VotingClassifier(**args_dict)
    else:
        print('classifier_name not supported')
        return
    if random_cv_iterations == -1:
        grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=verbose)
    else:
        grid_search = RandomizedSearchCV(estimator=clf, param_distributions=param_grid, n_iter=random_cv_iterations, cv=cv, n_jobs=-1, verbose=verbose)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_
    return best_params, best_score, best_model

def get_evaluation_repots(clf, X_test, y_test, as_dict=True, verbose=True):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=as_dict)
    if verbose:
        print(f'accuracy: {accuracy_score(y_test, y_pred)}')
        print(f'confusion_matrix: {confusion_matrix(y_test, y_pred)}')
        print(f'classification_report: {classification_report(y_test, y_pred)}')
    return acc, cm, cr

def preds_with_ids(gdf, clf ,drop_cols=None):
    if drop_cols == None:
        drop_cols = ['Location', 'geometry', 'Lat', 'Lon', 'ID', 'Target']
    if 'ID' not in gdf.columns:
        gdf['ID'] = 'test_gdf'
    X = gdf.drop(columns=drop_cols)
    preds = clf.predict(X)
    preds_with_ids = gdf.copy()
    preds_with_ids['Target'] = preds
    preds_with_ids = preds_with_ids[['ID', 'Target', 'geometry', 'Lat', 'Lon']]
    return preds_with_ids

def gdf_column_to_one_band_array(gdf, column_name):
    gdf = gdf.sort_values(by=['Lat', 'Lon'])
    gdf = gdf.reset_index(drop=True)
    unique_lats_count = gdf['Lat'].nunique()
    unique_lons_count = gdf['Lon'].nunique()
    rows_arr = [[] for i in range(unique_lats_count)]
    column_values = gdf[column_name].values
    for i in tqdm(range(len(column_values))):
        row_index = i // unique_lons_count
        rows_arr[row_index].append(column_values[i])
    rows_arr = np.array(rows_arr)
    return rows_arr

def true_color_df_to_image_array(true_color_df):
    x_axis = true_color_df['Lon'].unique().tolist()
    y_axis = true_color_df['Lat'].unique().tolist()
    rows_arr_B04 = gdf_column_to_one_band_array(true_color_df, 'B04')
    rows_arr_B03 = gdf_column_to_one_band_array(true_color_df, 'B03')
    rows_arr_B02 = gdf_column_to_one_band_array(true_color_df, 'B02')
    final_arr = np.dstack((rows_arr_B04, rows_arr_B03, rows_arr_B02))
    return final_arr, x_axis, y_axis

def targets_scatter_positions(gdf, x_axis, y_axis):
    lons_src = np.array(x_axis)
    lats_src = np.array(y_axis)
    lons_dst = np.array(gdf['Lon'])
    lats_dst = np.array(gdf['Lat'])
    ids_dst = np.array(gdf['ID'])
    src_indices = []
    dst_indices = []
    i=0
    ids_matched = []
    for lon_dst, lat_dst in zip(lons_dst, lats_dst):
        lon_dst_index = np.where(lons_src == lon_dst)
        lat_dst_index = np.where(lats_src == lat_dst)
        if len(lon_dst_index[0]) == 0 or len(lat_dst_index[0]) == 0:
            i += 1
            continue
        lon_dst_index = lon_dst_index[0][0]
        lat_dst_index = lat_dst_index[0][0]
        src_indices.append(i)
        dst_indices.append((lon_dst_index, lat_dst_index))
        ids_matched.append(ids_dst[i])
        i += 1
    targets =gdf['Target'].values
    targets = targets[src_indices]
    x_indices = [i[0] for i in dst_indices]
    y_indices = [i[1] for i in dst_indices]
    return x_indices, y_indices, targets, ids_matched

def save_best_model_and_create_submession(train_gdf, test_gdf, country, clf, classifer_key, do_save=False):
    X_train, X_test, y_train, y_test = get_train_test_splits_X_Y(train_gdf, test_size=0.99)
    acc, cm, cr = get_evaluation_repots(clf, X_test, y_test, verbose=True)
    print(f'accuracy: {acc}')
    print(f'confusion_matrix: {cm}')
    print(f'classification_report: {cr}')
    print('test_gdf and test_with_ids are not None')
    test_with_ids_preds = preds_with_ids(test_gdf, clf)
    submession_name= f'{country}_{classifer_key}_submession.csv'
    classifier_path = f'{country}_{classifer_key}_classifier.joblib'
    if do_save:
        print(f'Saving classifier to: {classifier_path}')
        print(f'Saving submession: {submession_name}')
        joblib.dump(clf, classifier_path)
        test_with_ids_preds.to_csv(submession_name, index=False)
    return test_with_ids_preds

def get_masking_dict_for_src_from_target(src_geom, target_gdf,tolarance = 8e-5):
    def try_again(target_point, n=1):
        new_tol = tolarance + (n*5e-5)
        i = 0
        for src_point in src_geom:
            if src_point.equals_exact(target_point, new_tol):
                return i
            i+=1
        return -1


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
            if src_point.equals_exact(target_point, tolarance):
                eq_points_indicies.append(i)
                eq_points_targets.append(t)
                target_point_matched = True
                break
            i+=1
        if not target_point_matched:
            n_vals = [i for i in range(1, 10000)]
            found_i = -1
            for n in n_vals:
                found_i = try_again(target_point, n)
                if found_i != -1:
                    break
            if found_i != -1:
                eq_points_indicies.append(found_i)
                eq_points_targets.append(t)
            else:
                print('not found')
                return
    calculated_dict = {
        'eq_points_indicies': eq_points_indicies,
        'eq_points_targets': eq_points_targets,
    }
    return calculated_dict

def train_and_test_gdfs_matched_to_original_train_test(train_gdf, test_gdf, train_original, test_original, country=None, use_cached=True):
    if country is not None and use_cached:
        cached_path = f'{country}_train_test_gdfs_matched_to_original_train_test.joblib'
        if os.path.exists(cached_path):
            print(f'Loading from cached_path: {cached_path}')
            return joblib.load(cached_path)
    test_original_claculated_dict = get_masking_dict_for_src_from_target(test_gdf.geometry.values, test_original,tolarance = 1e-8)
    train_original_claculated_dict = get_masking_dict_for_src_from_target(train_gdf.geometry.values, train_original,tolarance = 1e-8)
    eq_inds = train_original_claculated_dict['eq_points_indicies']
    eq_targets = train_original_claculated_dict['eq_points_targets']
    train_gdf = train_gdf.iloc[eq_inds]
    train_gdf['Target'] = eq_targets
    train_gdf = train_gdf.reset_index(drop=True)
    eq_inds = test_original_claculated_dict['eq_points_indicies']
    eq_targets = test_original_claculated_dict['eq_points_targets']
    test_gdf = test_gdf.iloc[eq_inds]
    test_gdf['Target'] = eq_targets
    test_gdf = test_gdf.reset_index(drop=True)
    train_gdf['ID'] = train_original['ID']
    test_gdf['ID'] = test_original['ID']

    if country is not None and use_cached:
        print(f'Saving to cached_path: {cached_path}')
        joblib.dump((train_gdf, test_gdf), cached_path) 

    return train_gdf, test_gdf

def train_on_all_samples(train_gdf, classifer_key='RF'):
    classifiers_dict = get_calssifiers_dict()
    drop_cols = ['Location', 'geometry', 'Lat', 'Lon', 'Target', 'ID']
    X_train = train_gdf.drop(columns=drop_cols)
    y_train = train_gdf['Target']
    args_dict = classifiers_dict[classifer_key]['args_dict']
    param_grid = classifiers_dict[classifer_key]['param_grid']
    classifier_name = classifiers_dict[classifer_key]['classifier_name']
    cv = classifiers_dict[classifer_key]['cv']
    print('training...')
    print('classifier_name:', classifier_name)
    print('args_dict:', args_dict)
    print('param_grid:', param_grid)
    print('cv:', cv)
    print('X_train.shape:', X_train.shape)
    best_params, best_score, best_model = grid_search(X_train, y_train, args_dict, param_grid, cv, classifier_name)
    # best_params, best_score, best_model = train_with_pipline(X_train, y_train)
    classifiers_dict[classifer_key]['best']={}
    classifiers_dict[classifer_key]['best']['param']=best_params
    classifiers_dict[classifer_key]['best']['acc']=best_score
    classifiers_dict[classifer_key]['best']['model']=best_model
    print(f'best_params: {best_params}')
    print(f'best_score: {best_score}')
    print(f'best_model: {best_model}')
    trained_classifier_dict = classifiers_dict[classifer_key]
    return trained_classifier_dict

def create_submession_ready_df_from_all_countries_preds():
    countries = ['Afghanistan', 'Sudan', 'Iran']
    submession_names = [f'{country}_submession.csv' for country in countries]
    submession_dfs_list = []
    for country, submession_name in zip(countries, submession_names):
        df = pd.read_csv(submession_name)
        submession_dfs_list.append(df)
    submession_df = pd.concat(submession_dfs_list, axis=0)
    submession_df = submession_df.reset_index(drop=True)
    test = utils.read_test_data()
    test_ids = test['ID'].values
    sort_indices = np.argsort(test_ids)
    submession_df = submession_df.iloc[sort_indices]
    submession_df = submession_df.reset_index(drop=True)
    drop_cols = ['geometry', 'Country', 'Lat', 'Lon']
    submession_df_ready = submession_df.copy()
    for col in drop_cols:
        if col in submession_df_ready.columns:
            submession_df_ready = submession_df_ready.drop(col, axis=1)
    if len(submession_df_ready.columns) == 2:
        if 'ID' in submession_df_ready.columns:
            if 'Target' in submession_df_ready.columns:
                submession_df_ready.to_csv('submession_df_ready.csv', index=False)
                print('Saved file: submession_df_ready.csv')
                return submession_df_ready
    print('submession_df_ready not saved because of wrong columns')
    print('submession_df_ready.columns:', submession_df_ready.columns)
    return submession_df_ready

def get_processed_data_for_country(country):
    porcessed_data_dict = get_porcessed_data_dict()
    train_gdf = porcessed_data_dict[country]['train']
    test_gdf = porcessed_data_dict[country]['test']
    train_original, test_original = get_orginal_train_test(country)
    print(f'train_gdf: {train_gdf.shape}, test_gdf: {test_gdf.shape}')
    print(f'train_original: {train_original.shape}, test_original: {test_original.shape}')
    # train_gdf, test_gdf = train_and_test_gdfs_matched_to_original_train_test(train_gdf, test_gdf, train_original, test_original)
    print(f'train_gdf: {train_gdf.shape}, test_gdf: {test_gdf.shape}')
    #drop cols with NaNs
    train_gdf = train_gdf.dropna(axis=1)
    test_gdf = test_gdf.dropna(axis=1)
    train_gdf = train_gdf.reset_index(drop=True)
    test_gdf = test_gdf.reset_index(drop=True)
    return train_gdf, test_gdf, train_original, test_original

def main(country, classifer_key='RF'):
    if country in ['Afghanistan', 'Iran', 'Sudan']:
        train_gdf, test_gdf, train_original, test_original = get_processed_data_for_country(country)
        if 'ID' not in train_gdf.columns:
            train_gdf['ID'] = 'train_gdf'
    # train_gdf = add_ndvi_cols(train_gdf)
    # test_gdf = add_ndvi_cols(test_gdf)
    # train_gdf = add_arvi(train_gdf)
    # test_gdf = add_arvi(test_gdf)
    # #drop cols with NaNs
    # train_gdf = train_gdf.dropna(axis=1)
    # test_gdf = test_gdf.dropna(axis=1)
    # train_gdf = train_gdf.reset_index(drop=True)
    # test_gdf = test_gdf.reset_index(drop=True)
    # train_gdf = normalize_df_by_mean_and_std(train_gdf)
    # test_gdf = normalize_df_by_mean_and_std(test_gdf)
    # cols_train = train_gdf.columns
    # cols_train_drop = [col for col in cols_train if '04-18' in col]
    # train_gdf = train_gdf.drop(columns=cols_train_drop)
    # cols_test = test_gdf.columns
    # cols_test_drop = [col for col in cols_test if '04-18' in col]
    # test_gdf = test_gdf.drop(columns=cols_test_drop)
    # train_gdf = train_gdf.reset_index(drop=True)
    # test_gdf = test_gdf.reset_index(drop=True)
    # print(f'train_gdf: {train_gdf.shape}, test_gdf: {test_gdf.shape}')
    trained_classifier_dict = train_on_all_samples(train_gdf, classifer_key= classifer_key)
    best_model = trained_classifier_dict['best']['model']
    test_with_ids_preds = save_best_model_and_create_submession(train_gdf, test_gdf, country,best_model, classifer_key, do_save=True)
    submession_name= f'{country}_submession.csv'
    test_with_ids_preds.to_csv(submession_name, index=False)
    print(f'Saved submession: {submession_name}')
    submession_df_ready = create_submession_ready_df_from_all_countries_preds()
    print(f'submession_df_ready: {submession_df_ready.shape}')
    print(submession_df_ready.Target.value_counts())
    print('Done!')


if __name__ == '__main__':
    #read country numnber from command line
    countries = ['Afghanistan', 'Iran', 'Sudan']
    classifer_keis = ['RF', 'AB', 'GB', 'NB', 'MLP', 'STACK', 'VOTE']
    print('============== Countries ==============')
    for i, country in enumerate(countries):
        print(f'{i}: {country}')
    print('============== Classifers ==============')
    for i, classifer_key in enumerate(classifer_keis):
        print(f'{i}: {classifer_key}')
    if len(sys.argv) > 1:
        country_number = sys.argv[1]
        country_number = int(country_number)
        if len(sys.argv) > 2:
            classifer_key_number = sys.argv[2]
            classifer_key_number = int(classifer_key_number)
        else:
            print('classifer_key_number not provided, rasied error')
            raise ValueError
    else:
        print('country_number not provided, rasied error')
        raise ValueError
    country = countries[country_number]
    classifer_key = classifer_keis[classifer_key_number]
    print(f'Country: {country}')
    print(f'classifer_key: {classifer_key}')
    main(country, classifer_key=classifer_key)
    create_submession_ready_df_from_all_countries_preds()
