import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import shapely.geometry

def read_data(filepath = 'train.csv'):
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




def main():
    train = read_data('Test.csv')
    st.write(train.head())
    m = folium.Map(location=[30, 30], zoom_start=2)
    countries = train['Country'].unique()
    for country in countries:
        train_country = train[train['Country'] == country]
        total_bounds = train_country.total_bounds
        bounds = [[total_bounds[1], total_bounds[0]], [total_bounds[3], total_bounds[2]]]
        train_country.explore(name = country, m=m)
        folium.Rectangle(bounds, color='red', name = country+' bounds').add_to(m)

    #add google satellite layer
    folium.TileLayer(
        tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr = 'Esri',
        name = 'Esri Satellite',
        overlay = False,
        control = True).add_to(m)


    folium.LayerControl().add_to(m)
    st_folium(m)
main()