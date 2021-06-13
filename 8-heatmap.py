#!/usr/bin/env python
# coding: utf-8

# In[478]:


import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import folium
from folium.plugins import MarkerCluster

from folium import plugins
from argopy import IndexFetcher as ArgoIndexFetcher
import cartopy
import datetime


import branca
from folium.features import DivIcon

import shapely


# In[479]:


def get_geojson_grid(lat1, lat2, lon1, lon2, lat_stride=6, lon_stride=6):
    """Returns a grid of geojson rectangles, and computes the exposure in each section of the grid based on the vessel data.

    spacing: in degrees

    Parameters
    ----------
    upper_right: array_like
        The upper right hand corner of "grid of grids" (the default is the upper right hand [lat, lon] of the USA).

    lower_left: array_like
        The lower left hand corner of "grid of grids"  (the default is the lower left hand [lat, lon] of the USA).

    n: integer
        The number of rows/columns in the (n,n) grid.

    Returns
    -------

    list
        List of "geojson style" dictionary objects
    """

    # numerically, should be that lat2>lat1, lon2>lon1

    geo_json_boxes = [] 

    lat_steps = np.linspace(lat1, lat2, int((lat2-lat1)/lat_stride)+1)
    lon_steps = np.linspace(lon1, lon2, int((lon2-lon1)/lon_stride)+1)
    print('num steps', int((lat2-lat1)/lat_stride))


    # NOTE: since we are working in the atlantic ocean
    # this returned grid will go 
    # from bottom to top (lat) and
    # from left to right (lon) 

    list_boxes_latlon = []

    for lat in lat_steps[:-1]:
        for lon in lon_steps[:-1]:
            # Define dimensions of box in grid
            #       lat+stride 
            #        |-----|
            # lon    |-----| lon + stride
            #          lat 
            upper_left = [lon, lat + lat_stride]
            upper_right = [lon + lon_stride, lat + lat_stride]
            lower_right = [lon + lon_stride, lat]
            lower_left = [lon, lat]

            # Define json coordinates for polygon
            # so each individual side, connecting order, clockwise
            coordinates = [
                upper_left,
                upper_right,
                lower_right,
                lower_left,
                upper_left
            ]

            lat_lon_box = {'lat': (lat, lat+lat_stride),
                           'lon': (lon, lon+lon_stride)}
            list_boxes_latlon.append(lat_lon_box)

            geo_json = {"type": "FeatureCollection",
                        "properties":{
                            "lower_left": lower_left,
                            "upper_right": upper_right
                        },
                        "features":[]}

            grid_feature = {
                "type":"Feature",
                "geometry":{
                    "type":"Polygon",
                    "coordinates": [coordinates],
                }
            }

            geo_json["features"].append(grid_feature)

            geo_json_boxes.append(geo_json)


    return geo_json_boxes, list_boxes_latlon


# In[480]:


# https://scitools.org.uk/cartopy/docs/latest/matplotlib/advanced_plotting.html
# https://matplotlib.org/basemap/api/basemap_api.html


# # Check for intersections

import warnings
#warnings.filterwarnings(action='once')
warnings.filterwarnings('ignore')


def create_line(row):
    line_feature =  { "type": "Feature",
                     "geometry": { "type": "LineString",
                                  "coordinates": row.line_coords, },
                     "properties": { "times": row.line_times,
                                    "style": { "color": 'black', "weight": 3},
                                    'icon': 'circle',
                                    'iconstyle':{
                                        'fillColor': color,
                                        'opacity': 0.9,
                                        'stroke': False,
                                        'radius':5
                                    },
                                    # Add click-to-popup
                                    'popup': f'Buoy ID: {row.wmo}, \n Date: {row.date}',#, Date uploaded: {row.date_update}',
                                    },
                     }
    return line_feature




def get_data(limits):
    index_loader = ArgoIndexFetcher(cache=True)
    region_and_time = [limits['lon1'], limits['lon2'], limits['lat1'], limits['lat2'], '2020-06-01', '2021-07'] # up to june (exclusive last bound)
    idx = index_loader.region(region_and_time)

    df_buoy = idx.to_dataframe()
    df_buoy.to_pickle(f'argoindex_{region_and_time}.pkl')

    # Clean out ones w/o lat lon
    df_buoy = df_buoy.dropna(subset=['latitude', 'longitude'])

    print('!--- Sanity checks for data retrieved ---!')
    list_buoy_ids = df_buoy.wmo.unique()
    print('list of buoys retrieved ', list_buoy_ids, 'from ', region_and_time)
    print('num retrieved: ', len(list_buoy_ids))
    return df_buoy 





### 
def heatmap_profile_counts(df_buoy, list_boxes_latlon, time_start, time_end):
    #lat_lon_box = {'lat': (lat, lat+lat_stride),
    #'lon': (lon, lon+lon_stride)}
    #list_boxes_latlon.append(lat_lon_box)

    df  = df_buoy[ (df_buoy.date >= time_start) & (df_buoy.date <= time_end)]
    print('time limited df', df.shape, time_start, time_end)


    zeros = [] # used for AABB(axis aligned bounding box) line intersection check later
    profile_counts_grid = []

    for latlon_box in list_boxes_latlon:
        lat1, lat2 = latlon_box['lat']
        lon1, lon2 = latlon_box['lon']

        itfits = df[
            (df['latitude'] >= lat1 ) &
            (df['latitude'] < lat2 ) &
            (df['longitude'] >= lon1 ) &
            (df['longitude'] < lon2)
        ]
        #print(itfits.shape[0], (lat1, lon1), (lat2, lon2))

        box = shapely.geometry.box(lat1, lon1, lat2, lon2)

        profile_counts_grid.append((itfits.shape[0], box))
        if itfits.shape[0] == 0: 
            zeros.append(box)

    print('\nitfits 3 examples: ', profile_counts_grid[:3], '\n')
    return profile_counts_grid, zeros 



#================
TIME_START = '2021-01-01'
TIME_END = '2021-02-28'
#TIME_END = '2021-01-31'

LIMITS = { 'lat1': 12, 'lat2': 12+30, 'lon1': -63, 'lon2': -63+45 }

df_buoy =  get_data(LIMITS)
print('df_buoy cols', df_buoy.columns)
#================

geo_json_boxes, list_boxes_latlon = get_geojson_grid( LIMITS['lat1'], LIMITS['lat2'], LIMITS['lon1'], LIMITS['lon2'], lat_stride=3, lon_stride=3 )

profile_counts_grid, zeros = heatmap_profile_counts(df_buoy,
                                                    list_boxes_latlon,
                                                    TIME_START, TIME_END)

#=============
#=============

def add_heatgrid_to_map(m, profile_counts_grid, geo_json_boxes):
    colormap = branca.colormap.linear.YlOrRd_09.scale(0, 30)
    colormap = colormap.to_step(index=[0, 1, 2, 5, 10, 15, 20, 25, 30])
    colormap.caption = 'Number of profiles'
    colormap.add_to(m)

    print(profile_counts_grid[0])
    print(geo_json_boxes[0])
    for (counts, shapely_box), geo_json in zip(profile_counts_grid,
                                               geo_json_boxes):

        if counts == 0:
            color = 'b'
            color = matplotlib.colors.to_hex(color)
        else:
            #color = plt.cm.Oranges(counts / len(boxes))
            #color = mpl.colors.to_hex(color)
            color = colormap(counts)

        gj = folium.GeoJson(geo_json,
                            style_function=lambda feature,
                            color=color: {
                                'fillColor': color,
                                'color':"black",
                                'weight': 0.4,
                                'dashArray': '5, 5',
                                'fillOpacity': 0.60,
                            })
        #print('adding to heatmap', counts)
        popup = folium.Popup(f"Counts:{counts}")
        gj.add_child(popup)

        m.add_child(gj)

    folium.map.Marker(
        [42,- 18],
        icon=DivIcon(
            icon_size=(150,36),
            icon_anchor=(0,0),
            html=f'<div style="font-size: 12pt"><b>Heatmap</b><br>Num. of profiles per 3°x3°<br>From <font color="purple">{TIME_START}</font> to '
            f'<font color="purple">{TIME_END}</font><br><font color="blue">Blue is 0 counts</font> </div>',
        )
    ).add_to(m)
    return m

#=============

m = folium.Map(zoom_start = 2, location=[30, -60]) #width=800, height=400)
m = add_heatgrid_to_map(m, profile_counts_grid, geo_json_boxes)

#=============

def add_buoy_to_map(m, df_buoy, time_start, time_end):

    # Format latlon and times for plotting
    df = df_buoy[['date', 'wmo']]
    df['longlat'] = list(zip(df_buoy.longitude, df_buoy.latitude))
    df  = df[ (df.date >= time_start) & (df.date <= time_end)]

    features = []

    list_buoy_ids = df.wmo.unique()
    num_colors = len(list_buoy_ids)
    cmap = matplotlib.cm.get_cmap('prism', num_colors)

    NaT_fill = 0
    gps_fill = -999


    #for ith_buoy, buoy in enumerate(list_buoy_ids):
    for ith_buoy, buoy in enumerate(list_buoy_ids[:2]):
        # Create timelapse lines for a single buoy
        buoy_df = df[ (df.wmo == buoy) ]

        print('i', ith_buoy, 'buoy id', buoy, 'num profiles',
              buoy_df.shape[0])

        buoy_df = buoy_df.sort_values(by=['date'])
        #print(buoy_df)

        # Create lines consist of # (latlong_pointA), timeA to (latlong_pointB, timeB)
        buoy_df['line_coords'] = list(zip(
            buoy_df['longlat'].shift(fill_value = gps_fill),
            buoy_df['longlat']))

        buoy_df['str_date'] = buoy_df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

        buoy_df['line_times'] = list(zip(
            buoy_df['str_date'].shift(fill_value = NaT_fill) ,
            buoy_df['str_date']
        ))

        for _, row in buoy_df.iterrows():
            print('This buoy is ', ith_buoy, row.wmo,  row.line_times, row.line_coords)
            if row.line_times[0] == 0:
                # Due to use of pd.shift, first 'line' is just point and  has no start time
                continue
            color = matplotlib.colors.rgb2hex(cmap(ith_buoy/num_colors))
            feature =  { "type": "Feature",
                     "geometry": { "type": "LineString",
                                  "coordinates": row.line_coords, },
                     "properties": { "times": row.line_times,
                                    "style": { "color": color, "weight": 3},
                                    'icon': 'circle',
                                    'iconstyle':{
                                        'fillColor': color,
                                        'opacity': 0.9,
                                        'stroke': False,
                                        'radius':5
                                    },
                                    # Add click-to-popup
                                    'popup': f'Buoy ID: {row.wmo}, \n Current Date: {row.date} \n\n'
                                    f'<i>Line Details: {list(zip(row.line_times, row.line_coords))}</i>',#, Date uploaded: {row.date_update}',
                                    },
                     }
            features.append(feature)
    #features
    print('num of features (aka number of buoys in timeframe)', len(features))

    plugins.TimestampedGeoJson(
        { "type": "FeatureCollection", "features": features},
        period="P1D", add_last_point=True,
        duration = 'P3M',
        min_speed = 1,
        max_speed = 10,
        transition_time = 300, # in millisec
    ).add_to(m)

    return m


m = add_buoy_to_map(m, df_buoy, TIME_START, TIME_END)

m.save(f'3x3_heatmap_with_buoylines-{LIMITS}-{TIME_START}-{TIME_END}.html')

