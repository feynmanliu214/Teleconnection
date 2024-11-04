#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 20:10:28 2023

@author: feynmanliu
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import pearsonr
from scipy.stats import t
import cftime
import xarray as xr
from sklearn.linear_model import LinearRegression



def index_to_year_month(start_year, index):
    """
    Convert a time index to a corresponding year and month.
    
    Parameters:
    - start_year: The starting year of the dataset (e.g., 1979)
    - index: The time index starting from 1
    
    Returns:
    - Tuple containing the year and month
    """
    
    # Calculate the year and month
    year = start_year + index // 12
    month = (index % 12) + 1
    
    return year, month

# Function to convert year and month to index
def year_month_to_index(start_year, year, month):
    """
    Convert a year and month to the corresponding time index.
    
    Parameters:
    - start_year: The starting year of the dataset (e.g., 1979)
    - year: The target year
    - month: The target month
    
    Returns:
    - The time index starting from 1
    """
    # Calculate the index
    index = ((year - start_year) * 12) + month - 1  # -1 because index is 0-based
    
    return index


def plot_data(time_index, ds, column_name, title):
    """
    Plot  data for a specific time index.
    
    Parameters:
    - time_index: integer, representing the month index starting from 0
    
    Returns:
    - A plot
    """

    # Convert time_index to year and month
    year, month = index_to_year_month(1979, time_index)
    
    # Extract the data for Z200 for the specific time index
    ds_data = ds[column_name][time_index, :, :].values
    lon = ds['lon'][:].values
    lat = ds['lat'][:].values
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    mesh = ax.pcolormesh(lon, lat, ds_data, shading='auto', cmap='viridis', transform=ccrs.PlateCarree())
    plt.colorbar(mesh, label='Height (m)', ax=ax)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Average {title} for {year}/{month}')
    
    # Add coastlines for continent boundaries
    ax.add_feature(cfeature.COASTLINE)
    
    plt.show()

def print_dataframe_basic_information(file_path):
    # Load the NetCDF file into an xarray Dataset
    dataframe = xr.open_dataset(file_path, decode_times=False)
    
    
    # Print general information about the Dataset
    print("Dataset Info:")
    print(dataframe)
    
    # Print dimensions information
    print("\nDimensions:")
    for dim, size in dataframe.dims.items():
        print(f"{dim}: {size}")
    
    # Print variable information
    print("\nVariables:")
    for var in dataframe.variables:
        print(f"{var}: {dataframe[var].shape}, {dataframe[var].dtype}")
    
    # Close the Dataset
    dataframe.close()


def extract_column(time_index, ds, column_name):
    """
    Return the z200 data for a specific time index.
    
    Parameters:
    - time_index: integer index to query
    - ds: xarray Dataset with a z200 variable
    
    Returns:
    - z200 data array for the specified time index.
    """
    # Assuming z200 is a variable in the dataset
    z200_data = ds[column_name][time_index, :, :].values
    return z200_data


def extract_consecutive_months(ds, column_name, selected_month, num_months, start_year=1979):
    # Define a dictionary to map month names to their index values
    month_to_index = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
    }

    # Convert the selected month to its index value
    selected_month_index = month_to_index[selected_month]

    # Initialize an array to store data for each month
    data_months = [[] for _ in range(num_months)]

    # Loop through the years
    for year in range(start_year + 1, 2024):
        # Calculate the index for the selected month
        index_selected_month = year_month_to_index(start_year, year, selected_month_index)

        # Loop to extract data for the selected month and the preceding months
        for i in range(num_months):
            month_index = index_selected_month - i
            data_months[i].append(extract_column(month_index, ds, column_name))

    # Convert the lists of arrays to 3D NumPy arrays
    data_months = [np.stack(month_data, axis=0) for month_data in data_months]

    return data_months
    

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

    # Find the closest index values in the latitude and longitude arrays
    lat_indices = np.where((ds.lat >= lat_south) & (ds.lat <= lat_north))[0]
    lon_indices = np.where((ds.lon >= lon_west) & (ds.lon <= lon_east))[0]

    # Slice the dataset using the found indices
    ds_subset = ds.isel(lat=lat_indices, lon=lon_indices)

    return ds_subset


def plot_continent_boundary(lat_south, lat_north, lon_west, lon_east):
    """
    Plots the boundary of North America based on the provided latitude and longitude limits.

    Parameters:
    - lat_south: Southern latitude limit.
    - lat_north: Northern latitude limit.
    - lon_west: Western longitude limit (adjusted for the dataset).
    - lon_east: Eastern longitude limit (adjusted for the dataset).
    """

    # Create a figure with an appropriate size
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Add coastlines, borders, and gridlines
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    
    # Set the extent based on the input parameters
    # Note that we need to convert the longitude to standard -180 to 180 if necessary
    lon_west_standard = lon_west if lon_west <= 180 else lon_west - 360
    lon_east_standard = lon_east if lon_east <= 180 else lon_east - 360
    ax.set_extent([lon_west_standard, lon_east_standard, lat_south, lat_north], crs=ccrs.PlateCarree())
    
    # Set title and labels
    ax.set_title('Boundary of North America')
    plt.show()


def linear_detrend_combine_data(combined_data):
    # Create a new array to hold the detrended values
    shape_values_winter_detrend = np.zeros_like(combined_data)

    # Iterate over each latitude and longitude to detrend the time series
    for lat in range(combined_data.shape[1]):
        for lon in range(combined_data.shape[2]):
            # Extract the time series for this lat/lon location
            time_series = combined_data[:, lat, lon]

            # Filter out NaN values and their corresponding years
            valid_indices = ~np.isnan(time_series)
            valid_years = np.arange(combined_data.shape[0])[valid_indices]
            valid_time_series = time_series[valid_indices]

            # Check if we have enough data to fit the linear regression
            if valid_time_series.size >= 3:
                # Reshape the valid time series for linear regression
                valid_time_series = valid_time_series.reshape(-1, 1)
                valid_years = valid_years.reshape(-1, 1)

                # Create and fit the linear regression model
                model = LinearRegression()
                model.fit(valid_years, valid_time_series)

                # Predict the trend for all years
                all_years = np.arange(combined_data.shape[0]).reshape(-1, 1)
                trend = model.predict(all_years)

                # Detrend the time series
                detrended = time_series - trend.flatten()
                detrended[~valid_indices] = np.nan  # Keep NaNs where they were originally
                shape_values_winter_detrend[:, lat, lon] = detrended
            else:
                # Set the detrended values for this lat/lon to NaN
                shape_values_winter_detrend[:, lat, lon] = np.nan

    # Return the detrended values
    return shape_values_winter_detrend


def reshape_and_scaled(X, Y, Year = 44):
    shape_1 = np.shape(X)[1]
    shape_2 = np.shape(X)[2]
    
    shape_3 = np.shape(Y)[1]
    shape_4 = np.shape(Y)[2]
    
    # Reshape the sliced data into 2D form: (time, latitude*longitude)
    X_2D = X.reshape(Year, shape_1*shape_2)
    Y_2D = Y.reshape(Year, shape_3*shape_4)
    
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_2D)
    Y_scaled = scaler.fit_transform(Y_2D)

    return X_scaled, Y_scaled
    

def maximal_covariance_analysis(X, Y, Year = 44):
    X = X.T
    Y = Y.T
    
    # Compute the cross-covariance matrix Cxy
    Cxy = np.dot(X, Y.T) / (Year - 1)

    # Perform SVD on the cross-covariance matrix
    U, sigma, Vt = np.linalg.svd(Cxy, full_matrices=False)

    # Scaling the singular values to represent the fraction of squared covariance
    sigma2 = sigma**2
    scf = sigma2 / np.sum(sigma2)

    # Plot the cumulative fraction of squared covariance explained by each mode
    plt.figure()
    plt.plot(scf[:30] * 100, 'x')
    plt.xlabel('MCA mode')
    plt.ylabel('Squared covariance fraction (%)')
    plt.title('Fraction of Squared Covariance Explained by Each MCA Mode')
    plt.show()

    # The left singular vectors are the patterns in X
    patterns_X = U

    # The right singular vectors are the patterns in Y
    patterns_Y = Vt.T

    return patterns_X, patterns_Y, sigma, scf

def calculate_time_series(X, Y, A, B, num_modes = 10, Year = 44):
    
    # Initialize T1 and T2
    T1 = np.zeros((X.shape[0], num_modes))
    T2 = np.zeros((Y.shape[0], num_modes))

    # Calculate T1 and T2 for each mode
    for k in range(num_modes):
        T1[:, k] = np.dot(Y, A[:, k])
        T2[:, k] = np.dot(Y, B[:, k])
    
    return T1, T2

def confidence_correlation(degrees_of_freedom, confidence_level=0.95):
    """
    Calculate the correlation coefficient for a given degree of freedom and confidence level.

    Args:
    degrees_of_freedom (int): The degrees of freedom.
    confidence_level (float): The confidence level, default is 0.95 (95%).

    Returns:
    float: The correlation coefficient.
    """
    # Calculate the critical t-value
    critical_t = t.ppf((1 + confidence_level) / 2, degrees_of_freedom)

    # Calculate the correlation coefficient
    correlation_coefficient = critical_t / ((degrees_of_freedom + critical_t**2)**0.5)

    return correlation_coefficient

def plot_mca_results(A, B, T1, T2,scf, shape, latitude, mode_index=0):
    """
    Plots the spatial patterns and time series for a given mode from MCA results.

    Parameters:
    A (numpy.ndarray): Pattern matrix for 'X' with a focus on a specific region.
    B (numpy.ndarray): Pattern matrix for 'Y' for the global domain.
    T1 (numpy.ndarray): Time series data for 'X'.
    T2 (numpy.ndarray): Time series data for 'Y'.
    year (list or numpy.ndarray): Array of years.
    mode_index (int): Index of the mode to be plotted. Default is 0 (first mode).
    """
    
    confidence_corr = confidence_correlation(43)

    # Modify the colormap
    # Create a modified coolwarm colormap with white at the center
    coolwarm = plt.cm.coolwarm
    coolwarm_mod = mcolors.LinearSegmentedColormap.from_list(
        "coolwarm_mod",
        [(x, coolwarm(x)) if x != 0.5 else (x, (1, 1, 1)) for x in np.linspace(0, 1, 256)],
        N=256
    )


    year = np.linspace(1979, 2022, 44)
    scf = scf*100
    
    shape_1, shape_2 = shape
    lat_south, lat_north, lon_west, lon_east = latitude
    
    # Reshape patterns according to your grid dimensions
    pattern_A = A[:, mode_index].reshape(shape_1, shape_2)
    pattern_B = B[:, mode_index].reshape(shape_1, shape_2)

    pattern_A = pattern_A*np.std(T1[:, mode_index])
    pattern_B = pattern_B*np.std(T2[:, mode_index])

    pattern_A_masked = np.ma.masked_inside(pattern_A, -confidence_corr, confidence_corr)
    
    # For pattern A
    total_area_A = np.prod(pattern_A.shape)
    area_beyond_confidence_A = np.count_nonzero(~pattern_A_masked.mask)
    percentage_area_beyond_A = (area_beyond_confidence_A / total_area_A) * 100


    # Latitude and longitude grid for pattern B
    lat_B = np.linspace(lat_south, lat_north, pattern_B.shape[0])
    lon_B = np.linspace(lon_west, lon_east, pattern_B.shape[1])  # Adjusted for the correct longitude range

    # Latitude and longitude grid for pattern A
    lat_A = lat_B
    lon_A = lon_B

    # Set up the map projection
    projection = ccrs.PlateCarree()

    # Plot for pattern A (X) 
    fig1, ax1 = plt.subplots(figsize=(10, 5), subplot_kw={'projection': projection})
    # Enhance coastline and borders
    ax1.add_feature(cfeature.COASTLINE, linewidth=1.0)
    ax1.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1.0)

    # Create a contourf plot (filled contour)
    contourf_A = ax1.contourf(lon_A, lat_A, pattern_A_masked, transform=ccrs.PlateCarree(), cmap='coolwarm')
    
    # Adding a color bar
    cbar = plt.colorbar(contourf_A, orientation='horizontal', pad=0.10, aspect=50)
    cbar.set_label('Covariance Strength')

    # Add title
    ax1.set_title(f'Spatial Pattern for Atmosphere Current Strength Mode {mode_index + 1} (SCF = {scf[mode_index]:.2f}%)')

    # Define a custom function to choose contour colors
    def get_contour_colors(level):
        if level > 0:
            return 'red'
        else:
            return 'blue'

    # Add contour lines with custom colors and reduced linewidth
#    contour_A = ax1.contour(lon_A, lat_A, pattern_A, levels=[-0.5, 0, 0.5],
#                            colors=[get_contour_colors(l) for l in [-0.5, 0, 0.5]],
#                            transform=ccrs.PlateCarree(), linewidths=0.3, linestyle = 'dashed')
#    ax1.clabel(contour_A, inline=True, fontsize=8)

    step_size_lon = 10
    step_size_lat = 10
    
    #ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)



    plt.show()
    
    pattern_B_masked = np.ma.masked_inside(pattern_B, -confidence_corr, confidence_corr)
    
    # For pattern B
    total_area_B = np.prod(pattern_B.shape)
    area_beyond_confidence_B = np.count_nonzero(~pattern_B_masked.mask)
    percentage_area_beyond_B = (area_beyond_confidence_B / total_area_B) * 100

    # Plot for pattern B (Y)
    fig2, ax2 = plt.subplots(figsize=(10, 5), subplot_kw={'projection': projection})
    # Enhance coastline and borders
    ax2.add_feature(cfeature.COASTLINE, linewidth=1.0)
    ax2.add_feature(cfeature.BORDERS, linestyle='-', linewidth=1.0)

    # Create a contourf plot (filled contour) for pattern B
    contourf_B = ax2.contourf(lon_B, lat_B, pattern_B_masked, transform=ccrs.PlateCarree(), cmap='coolwarm')
    # Adding a color bar
    
    cbar = plt.colorbar(contourf_B, orientation='horizontal', pad=0.10, aspect=50)
    cbar.set_label('Covariance Strength')

    # Add title
    ax2.set_title(f'Spatial Pattern for Atmosphere River Occurrence Density Mode {mode_index + 1} (SCF = {scf[mode_index]:.2f}%)')

    # Define a custom function to choose contour colors
    def get_contour_colors(level):
        if level > 0:
            return 'red'
        else:
            return 'blue'

    # Add contour lines with custom colors and reduced linewidth for pattern B
    #contour_B = ax2.contour(lon_B, lat_B, pattern_B, levels=[-0.5, 0, 0.5],
                            #colors=[get_contour_colors(l) for l in [-0.5, 0, 0.5]],
                            #transform=ccrs.PlateCarree(), linewidths=0.3, linestyle = 'dashed')
    #ax2.clabel(contour_B, inline=True, fontsize=8)

    #ax2.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)


    plt.show()

    # Standardize the time series
    T1_std = T1[:, mode_index] / np.std(T1[:, mode_index])
    T2_std = T2[:, mode_index] / np.std(T2[:, mode_index])

    # Compute the Pearson correlation coefficient
    corr, _ = pearsonr(T1_std, T2_std)

    # Plot for the time series
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.plot(year, T1_std, label='Current Strength Time Series')
    ax3.plot(year, T2_std, label='AR Density Time Series')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Normalized Time Series')
    ax3.legend()

    # Set title with Pearson correlation coefficient
    ax3.set_title(f'Standardized Amplitude Associated with MCA Mode {mode_index + 1} (SCF = {scf[mode_index]:.2f} %, r = {corr:.2f})')
    plt.show()
    
    # Sum of correlations and average for A
    sum_correlations_A = np.sum(np.abs(pattern_A_masked))
    average_correlation_A = sum_correlations_A / area_beyond_confidence_A

    # Sum of correlations and average for B
    sum_correlations_B = np.sum(np.abs(pattern_B_masked))
    average_correlation_B = sum_correlations_B / area_beyond_confidence_B
    
    return percentage_area_beyond_A, percentage_area_beyond_B, average_correlation_A, average_correlation_B

def execute_MCA_func(Matrix_1, Matrix_2, coordinate, detrended = False):

    #Extract the matrix to desired coordinate
    lat_south, lat_north, lon_west, lon_east = coordinate
    
    # Find the closest index values in the latitude and longitude arrays
    lat_indices = np.where((lat_values >= lat_south) & (lat_values <= lat_north))[0]
    lon_indices = np.where((lon_values >= lon_west) & (lon_values <= lon_east))[0]

    Matrix_1 = Matrix_1[:, lat_indices.min():lat_indices.max()+1, lon_indices.min():lon_indices.max()+1]
    
    Matrix_2 = Matrix_2[:, lat_indices.min():lat_indices.max()+1, lon_indices.min():lon_indices.max()+1]
    
    #detrend the data to cancel the effect of global warming
    if detrended:
        Matrix_1 = linear_detrend_combine_data(Matrix_1)
        Matrix_2 = linear_detrend_combine_data(Matrix_1)
    
    #scaled and reshape the data
    X_scaled, Y_scaled = reshape_and_scaled(Matrix_1, Matrix_2)
    
    #perfom the MCA
    A, B, s, scf = maximal_covariance_analysis(X_scaled, Y_scaled)
    
    #calculate the time series
    T1, T2 = calculate_time_series(X_scaled, Y_scaled, A, B)

        
    return  A, B, s, scf, T1, T2


def plot_correlation_matrix(df_1, T2, mode_index, boundary, title):

    T2_std = T2[:, mode_index] / np.std(T2[:, mode_index])
    
    shape_1 = np.shape(df_1)[1]
    shape_2 = np.shape(df_1)[2]
    
    # Initialize the Pearson correlation matrix
    correlation_matrix = np.zeros((shape_1, shape_2))
    
    # Define latitude and longitude ranges and steps
    lat_south, lat_north, lon_west, lon_east = boundary

    lat_step = (lat_north - lat_south) / (shape_1 - 1)  # Given the shape, it's 121 steps but 120 intervals
    lon_step = (lon_east - lon_west) / (shape_2 - 1)  # Same reasoning as for latitude
    
    # Generate latitude and longitude labels
    lat_labels = np.linspace(lat_south, lat_north, int(shape_1))
    lon_labels = np.linspace(lon_west, lon_east, int(shape_2))
    

    # Initialize the correlation matrix
    correlation_matrix = np.zeros((shape_1, shape_2))

    # Calculate Pearson correlation for each point
    for lat_idx in range(shape_1):
        for lon_idx in range(shape_2):
            series1 = df_1[:, lat_idx, lon_idx]
            series2 = T2_std

            # Identify indices where neither series has NaN values
            valid_indices = ~np.isnan(series1) & ~np.isnan(series2)

            # Extract the matched non-NaN elements from both series
            valid_series1 = series1[valid_indices]
            valid_series2 = series2[valid_indices]

            # Check if there are enough data points for correlation
            if len(valid_series1) >= 3 and len(valid_series2) >= 3:
                # Compute the Pearson correlation with valid data
                correlation_matrix[lat_idx, lon_idx], _ = pearsonr(valid_series1, valid_series2)
            else:
                # Set the correlation to NaN if there are not enough data points
                correlation_matrix[lat_idx, lon_idx] = np.nan

    # Mask the correlation matrix values within the critical correlation
    critical_correlation = confidence_correlation(degrees_of_freedom = 44 -2, confidence_level=0.95)
    correlation_matrix_masked = np.ma.masked_inside(correlation_matrix, -critical_correlation, critical_correlation)

    total_area = np.prod(correlation_matrix.shape)
    area_beyond_confidence = np.count_nonzero(~correlation_matrix_masked.mask)
    percentage_area_beyond = (area_beyond_confidence / total_area) * 100
    
    # Create a new matplotlib figure and set its size
    fig = plt.figure(figsize=(15, 10))
    
    # Create a cartopy map with longitude and latitude projection
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    
    # Plot the heatmap
    c = ax.pcolormesh(lon_labels, lat_labels, correlation_matrix_masked, cmap='coolwarm', shading='auto')
    
    # Add a color bar
    plt.colorbar(c, ax=ax, orientation='vertical', label='Correlation Coefficient')
    
    # Set plot titles and labels
    plt.title('Pointwise Pearson Correlation Coefficients' + title + " of Mode " + str(mode_index + 1))
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    plt.show()
    return percentage_area_beyond, np.nanmean(abs(correlation_matrix_masked))
