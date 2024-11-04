#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 20:39:14 2023

@author: feynmanliu
"""

import xarray as xr
import numpy as np

# Open the NetCDF file
ds_AR = xr.open_dataset("../data/AR-ERA5.nc")

shape = ds_AR.variables['shape']
year = ds_AR.variables['year']
month = ds_AR.variables['month']
day = ds_AR.variables['day']

Latitude = ds_AR.variables['lat']
Longitude = ds_AR.variables['lon']

time = ds_AR.variables['time']

def get_time_from_index(ds, index):
    """
    Return the date and hour for a given index from an xarray dataset's time dimension.
    
    Parameters:
    - ds: xarray Dataset with a time dimension
    - index: integer index to query
    
    Returns:
    - String in the format "YYYY-MM-DD HH"
    """
    if index < 0 or index >= len(ds['time']):
        raise ValueError("Index out of bounds for the time dimension.")
    
    # Extract the datetime64 object from the dataset
    timestamp = ds['time'].values[index]
    
    # Convert to a standard datetime object for string formatting
    dt = timestamp.astype('M8[m]').astype('O')
    
    # Return formatted string
    return dt.strftime('%Y-%m-%d %H')

def get_data_for_time_index(time_index):
    """
    Return the Latitude, Longitude, and Shape values for a given time index.
    
    Parameters:
    - time_index: integer index to query
    
    Returns:
    - Latitude, Longitude, and Shape arrays for the specified time index.
    """
    
    # Check if the provided index is within bounds
    if time_index < 0 or time_index >= time.shape[0]:
        raise ValueError("Time index out of bounds.")
    
    # Extract the Shape values for the given time index
    # Based on the provided shape (1, 64644, 1, 121, 240), we squeeze out singular dimensions
    shape_values = np.squeeze(shape[:, time_index, :, :, :])
    
    # Latitude and Longitude values can be directly returned as they don't depend on the time index
    return Latitude.values, Longitude.values, shape_values


def get_indices_for_month_year(ds, year, month):
    """
    Return indices of the time values that belong to a specific month and year.
    
    Parameters:
    - ds: xarray Dataset with a time dimension
    - year: desired year (e.g., 2020)
    - month: desired month (e.g., 3 for March)
    
    Returns:
    - List of indices
    """
    
    # Convert time values to standard datetime objects
    datetime_values = ds['time'].values.astype('M8[M]').astype('O')
    
    # Get boolean array where month and year match
    bool_array = [(dt.year == year and dt.month == month) for dt in datetime_values]
    
    # Extract indices where boolean array is True
    indices = [i for i, val in enumerate(bool_array) if val]
    
    return indices
    
def get_data_for_time_indices(time_indices):
    """
    Return the Latitude, Longitude, and Shape values for a list of time indices.
    
    Parameters:
    - time_indices: list of integer indices to query
    
    Returns:
    - Latitude, Longitude, and Shape arrays for the specified time indices.
    """
    
    # Check if all provided indices are within bounds
    if any(idx < 0 for idx in time_indices) or any(idx >= time.shape[0] for idx in time_indices):
        raise ValueError("One or more time indices are out of bounds.")
    
    # Extract the Shape values for all given time indices
    # For each index, the shape values are extracted and added to a list
    shape_list = [np.squeeze(shape[:, idx, :, :, :]) for idx in time_indices]
    
    # Combine the list of arrays into a single array with an extra dimension
    shape_values_combined = np.stack(shape_list, axis=0)
    
    # Latitude and Longitude values can be directly returned as they don't depend on the time indices
    return Latitude.values, Longitude.values, shape_values_combined

def process_shape_values(shape_values):
    """
    Process the shape values to calculate the average occurrence over the first dimension.
    
    Parameters:
    - shape_values: ndarray with shape (time, lat, lon)
    
    Returns:
    - Shape values average with shape (1, lat, lon).
    """
    
    # Replace NaN values with 0
    shape_values = np.nan_to_num(shape_values, nan=0)
    
    # Convert any non-zero value to 1
    shape_values[shape_values != 0] = 1
    
    # Sum over the first dimension to get the total count of occurrences
    total_occurrences = np.sum(shape_values, axis=0)
    
    # Divide by the length of the first dimension to get the average
    shape_values_average = total_occurrences / shape_values.shape[0]
    
    # Add an extra first dimension to make it (1, lat, lon)
    shape_values_average = np.expand_dims(shape_values_average, axis=0)
    
    return shape_values_average


def calculate_monthly_average_atmospheric_rivers(ds, year, month):
    # Get indices for the specified month and year
    time_indices = get_indices_for_month_year(ds, year, month)
    
    # Get data for the obtained indices
    lat_values, lon_values, shape_values = get_data_for_time_indices(time_indices)
    
    # Process shape values to obtain the average
    shape_values_average = process_shape_values(shape_values)
    
    return shape_values_average















