import rasterio
import mask
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import numpy as np

def fix_image(img, no_brightness=False):
    def normalize(band):
        band_min, band_max = (band.min(), band.max())
        return ((band-band_min)/((band_max - band_min)))
    def brighten(band):
        alpha=0.13
        beta=0
        return np.clip(alpha*band+beta, 0,255)
    def gammacorr(band):
        gamma=2
        return np.power(band, 1/gamma)
    red   = img[:, :, 0]
    green = img[:, :, 1]
    blue  = img[:, :, 2]
    if no_brightness:
        red_b=red
        blue_b=blue
        green_b=green
    else:
        red_b=brighten(red)
        blue_b=brighten(blue)
        green_b=brighten(green)
    red_bg=gammacorr(red_b)
    blue_bg=gammacorr(blue_b)
    green_bg=gammacorr(green_b)
    red_bgn = normalize(red_bg)
    green_bgn = normalize(green_bg)
    blue_bgn = normalize(blue_bg)
    rgb_composite_bgn= np.dstack((red_bgn, green_bgn, blue_bgn))
    return rgb_composite_bgn



def get_tiff(date, location):
    parent_dir = f'./data/satellite_images/{date}/{location}/ALL/'
    files_in_dir = os.listdir(parent_dir)[0]
    tiff_path = f'{parent_dir}{files_in_dir}/response.tiff'
    rasterio_img = rasterio.open(tiff_path)
    bands_all = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08','B8A', 'B09', 'B10', 'B11', 'B12']
    df = mask.gdf_from_raster(rasterio_img, bands_all)
    true_color_columns = ['B04', 'B03', 'B02']
    true_color_columns.append('Lat')
    true_color_columns.append('Lon')
    true_color_columns.append('geometry')
    true_color_gdf = df[true_color_columns]
    return true_color_gdf

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


def split_target_scatter_dict_to_pos_neg(target_scatter_dict, positive_target=1, negative_target=0):
    targets = target_scatter_dict['targets']
    x_inds = target_scatter_dict['x_inds']
    y_inds = target_scatter_dict['y_inds']
    ids_matched = target_scatter_dict['ids_matched']
    x_inds_positive = []
    y_inds_positive = []
    targets_positive = []
    id_matched_positive = []
    x_inds_negative = []
    y_inds_negative = []
    targets_negative = []
    id_matched_negative = []
    for i in range(len(targets)):
        if targets[i] == positive_target:
            x_inds_positive.append(x_inds[i])
            y_inds_positive.append(y_inds[i])
            targets_positive.append(targets[i])
            id_matched_positive.append(ids_matched[i])
        if targets[i] == negative_target:
            x_inds_negative.append(x_inds[i])
            y_inds_negative.append(y_inds[i])
            targets_negative.append(targets[i])
            id_matched_negative.append(ids_matched[i])
    target_scatter_dict['x_inds_positive'] = x_inds_positive
    target_scatter_dict['y_inds_positive'] = y_inds_positive
    target_scatter_dict['targets_positive'] = targets_positive
    target_scatter_dict['id_matched_positive'] = id_matched_positive
    target_scatter_dict['x_inds_negative'] = x_inds_negative
    target_scatter_dict['y_inds_negative'] = y_inds_negative
    target_scatter_dict['targets_negative'] = targets_negative
    target_scatter_dict['id_matched_negative'] = id_matched_negative
    print(f'Number of positive targets: {len(targets_positive)}')
    print(f'Number of negative targets: {len(targets_negative)}')
    return target_scatter_dict

def scatter_plot_target_scatter_dict(target_scatter_dict, title, ax, sparse=False):
    x_inds_positive = target_scatter_dict['x_inds_positive']
    y_inds_positive = target_scatter_dict['y_inds_positive']
    x_inds_negative = target_scatter_dict['x_inds_negative']
    y_inds_negative = target_scatter_dict['y_inds_negative']

    y_inds_positive = [len(target_scatter_dict['y_axis']) - i for i in y_inds_positive]
    y_inds_negative = [len(target_scatter_dict['y_axis']) - i for i in y_inds_negative]

    if sparse:
        x_inds_positive = x_inds_positive[::10]
        y_inds_positive = y_inds_positive[::10]
        x_inds_negative = x_inds_negative[::10]
        y_inds_negative = y_inds_negative[::10]

    ax.scatter(x_inds_positive, y_inds_positive, s=10, label='Positive', color = 'red')
    for i in range(len(x_inds_positive)):
        ax.text(x_inds_positive[i], y_inds_positive[i], i , fontsize=30, color='red')
    ax.scatter(x_inds_negative, y_inds_negative, s=10, label='Negative', color = 'blue')
    for i in range(len(x_inds_negative)):
        ax.text(x_inds_negative[i], y_inds_negative[i], i , fontsize=30, color='blue')
    ax.legend(prop={'size': 30})
    ax.set_title(title)
    return ax
        
def main_gdf_with_target_over_images(gdf, dates, locations, fig_name='test.pdf'):
    if 'ID' not in gdf.columns:
        gdf['ID'] = gdf.index
    tiffs = {}
    dates = list(set(dates))
    locations = list(set(locations))
    print(f'Number of dates: {len(dates)}')
    print(f'Number of locations: {len(locations)}')
    print(f'Total number of images: {len(dates) * len(locations)}')
    print(f'Shape of gdf: {gdf.shape}')
    print()
    print('loading tiffs...')
    for date in dates:
        for location in locations:
            key = f'{date}_{location}_ALL_tiff'
            tiffs[key] = {}
            tiffs[key]['gdf_all'] = get_tiff(date, location)
    print('Generating image arrays...')
    for key in tiffs:
        true_color_arr, x_axis, y_axis = true_color_df_to_image_array(tiffs[key]['gdf_all'])
        tiffs[key]['true_color_arr'] = true_color_arr
        tiffs[key]['x_axis'] = x_axis
        tiffs[key]['y_axis'] = y_axis
    print('Generating scatter positions...')
    for key in tiffs:
        x_inds, y_inds, targets, ids_matched = targets_scatter_positions(gdf, tiffs[key]['x_axis'], tiffs[key]['y_axis'])
        print(f'Image {key} has {len(targets)} matched targets out of {len(gdf)}')
        tiffs[key]['x_inds'] = x_inds
        tiffs[key]['y_inds'] = y_inds
        tiffs[key]['targets'] = targets
        tiffs[key]['ids_matched'] = ids_matched
    for key in tiffs:
        tiffs[key] = split_target_scatter_dict_to_pos_neg(tiffs[key], positive_target=1, negative_target=0)
    num_rows = len(dates)
    num_cols = len(locations)
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(80,80))
    print('Generating plots...')
    if num_rows !=1 and num_cols != 1:
        #TODO: Add support for more than one row and column
        print('Not supported yet, please use only one row or one column')
        raise NotImplementedError
    else:
        key = f'{dates[0]}_{locations[0]}_ALL_tiff'
        ax.imshow(fix_image(tiffs[key]['true_color_arr']))
        title = f'{dates[0]}_{locations[0]}'
        ax = scatter_plot_target_scatter_dict(tiffs[key], title, ax)
    fig.savefig(fig_name)
    plt.tight_layout()
    plt.show()

    return tiffs


