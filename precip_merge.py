#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 17:06:21 2023

@author: feynmanliu
"""

import xarray as xr


import numpy as np
import os

def subset_dataset(ds, lat_south, lat_north, lon_west, lon_east):
    """
    Subset an xarray Dataset based on specified latitude and longitude boundaries.

    Parameters:
    ds (xarray.Dataset): The input dataset.
    lat_south (float): The southern latitude boundary.
    lat_north (float): The northern latitude boundary.
    lon_west (float): The western longitude boundary.
    lon_east (float): The eastern longitude boundary.

    Returns:
    xarray.Dataset: A subset of the original dataset within the specified region.
    """
    lat_indices = np.where((ds.lat >= lat_south) & (ds.lat <= lat_north))[0]
    lon_indices = np.where((ds.lon >= lon_west) & (ds.lon <= lon_east))[0]

    ds_subset = ds.isel(lat=lat_indices, lon=lon_indices)
    return ds_subset

def process_precip_files(folder_path, lat_south, lat_north, lon_west, lon_east, output_nc_path):
    """
    Process multiple NetCDF files, extract North American precipitation data, and merge into a single file.

    Parameters:
    folder_path (str): Path to the folder containing the NetCDF files.
    lat_south, lat_north, lon_west, lon_east: Geographic boundaries for North America.
    output_nc_path (str): Path to the output NetCDF file.
    """
    nc_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.nc')])
    combined_ds = None

    for nc_file in nc_files:
        file_path = os.path.join(folder_path, nc_file)
        
        with xr.open_dataset(file_path) as ds:
            ds_subset = subset_dataset(ds, lat_south, lat_north, lon_west, lon_east)
            
            if combined_ds is None:
                combined_ds = ds_subset
            else:
                combined_ds = xr.concat([combined_ds, ds_subset], dim='time')

    combined_ds.to_netcdf(output_nc_path)

# Define North American boundaries
lat_south = 15
lat_north = 60
lon_west = 190
lon_east = 310

# Usage
folder_path_1 = "../precip"
output_nc_path = "../data/precip_north_america.nc"
process_precip_files(folder_path_1, lat_south, lat_north, lon_west, lon_east, output_nc_path)


def calculate_monthly_average(input_nc_path, output_nc_path):
    """
    Calculate the monthly average from a daily NetCDF dataset.

    Parameters:
    input_nc_path (str): Path to the input NetCDF file with daily data.
    output_nc_path (str): Path to the output NetCDF file with monthly average data.
    """
    # Open the dataset
    with xr.open_dataset(input_nc_path) as ds:
        # Resample the data to monthly frequency and calculate the mean
        monthly_avg = ds.resample(time='1MS').mean()

        # Save the monthly averaged data to a new NetCDF file
        monthly_avg.to_netcdf(output_nc_path)

# Usage
input_nc_path = "../data/precip_north_america.nc"
output_nc_path = "../data/precip_north_america_monthly.nc"
calculate_monthly_average(input_nc_path, output_nc_path)
