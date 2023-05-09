# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 19:32:18 2023

@author: mfors
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle as pk
from scipy.stats import ks_2samp
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd

def find_true_sequences(bool_array, threshold):
    """
    Finds sequences of True values in a boolean array that are above a certain
    length (specified by the threshold parameter) and returns a pandas DataFrame
    containing the start point, end point, and length of each sequence.
    
    Parameters:
    bool_array (numpy array): The boolean array to search for true sequences.
    threshold (int): The minimum length of a true sequence to include in the output.
    
    Returns:
    pandas DataFrame: A DataFrame containing the start point, end point, and length
    of each true sequence found in the input boolean array.
    """
    
    # Find the indices where the boolean array changes from False to True
    starts = np.where(np.diff(np.concatenate(([False], bool_array.astype(int)))) == 1)[0]
    
    # Find the indices where the boolean array changes from True to False
    ends = np.where(np.diff(np.concatenate((bool_array.astype(int), [False]))) == -1)[0]
    
    # If the boolean array starts with True values, remove the first sequence
    if bool_array[0] == True:
        starts = starts[1:]
        ends = ends[1:]
    
    # If the boolean array ends with True values, remove the last sequence
    if bool_array[-1] == True:
        starts = starts[:-1]
        ends = ends[:-1]
    
    # Calculate the lengths of the true sequences
    lengths = ends - starts + 1
    
    # Filter out sequences that are shorter than the threshold
    mask = lengths >= threshold
    starts = starts[mask]
    ends = ends[mask]
    lengths = lengths[mask]
    
    # Create a pandas DataFrame with the results
    df = pd.DataFrame({"Start": starts, "End": ends, "Length": lengths})
    
    return df

def linear_regression_force_zero(x, y):
    """
    Fits a linear regression model to the input x and y values, with the constraint
    that the line must pass through the origin (i.e., y-intercept = 0), and returns
    the slope and R-squared value of the fit line.
    
    Parameters:
    x (numpy array): The x values.
    y (numpy array): The y values.
    
    Returns:
    slope (float): The slope of the fit line.
    r_squared (float): The R-squared value of the fit line.
    """
    
    # Reshape the input arrays into 2D arrays
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    
    # Fit the linear regression model with the constraint that the line must
    # pass through the origin
    model = LinearRegression(fit_intercept=False)
    model.fit(x, y)
    
    # Extract the slope of the fit line
    slope = model.coef_[0][0]
    
    # Calculate the R-squared value of the fit line
    y_pred = model.predict(x)
    r_squared = r2_score(y, y_pred)
    
    # Return the results
    return slope, r_squared

def ks_test(data1, data2, threshold=None):
    """
    Performs a two-sample Kolmogorov-Smirnov test on data1 and data2 to
    determine whether they are from the same distribution, using only
    values above a specified threshold (if provided).
    
    Parameters:
    data1 (numpy array): First data set.
    data2 (numpy array): Second data set.
    threshold (float, optional): Values below this threshold will be excluded
        from the analysis. Default is None (i.e., no thresholding).
    
    Returns:
    p-value (float): The p-value of the KS-test.
    D (float): The KS statistic of the KS-test.
    """
    
    # Apply the threshold (if provided)
    if threshold is not None:
        data1 = data1[data1 >= threshold]
        data2 = data2[data2 >= threshold]
    
    # Compute the KS statistic and p-value
    D, p_value = ks_2samp(data1, data2)
    
    # Return the results
    return p_value, D

with open('swim_kinematics_axolotl.pkl', 'rb') as tt:
    axo_data = pk.load(tt)
    
with open('swim_kinematics_xenopus.pkl', 'rb') as tt:
    xen_data = pk.load(tt)

plt.close('all')

f, ax = plt.subplots()
ax.plot(abs(xen_data['head angular acceleration']), xen_data['swim_velocity'], '.', color=[5/255, 190/255, 120/255])
ax.plot(abs(axo_data['head angular acceleration']), axo_data['swim_velocity'], '.', color=[242/255, 183/255, 124/255])
ax.set_xlabel('head angular acceleration (°/s/s)')
ax.set_ylabel('swim velocity (mm/s)')


f2, ax2 = plt.subplots()
ax2.plot(abs(xen_data['head angular acceleration']), xen_data['tail beat velocity'], '.', color=[5/255, 190/255, 120/255])
ax2.plot(abs(axo_data['head angular acceleration']), axo_data['tail beat velocity'], '.', color=[242/255, 183/255, 124/255])
ax2.set_xlabel('head angular acceleration (°/s/s)')
ax2.set_ylabel('tail beat velocity (°/s)')

f3, ax3 = plt.subplots()
n_bins = np.arange(0,100,2)
ax3.hist(xen_data['swim_velocity'], n_bins, density=True, color=[5/255, 190/255, 120/255])
ax3.hist(axo_data['swim_velocity'], n_bins, density=True, color=[242/255, 183/255, 124/255])
plt.xlabel('swim velocity (mm/s)')
plt.ylabel('Probability density')
p_swimvel, d_swimvel = ks_test(xen_data['swim_velocity'], axo_data['swim_velocity'],2)

f4, ax4 = plt.subplots()
n_bins = np.arange(0,100,2)
ax4.hist(xen_data['tail beat velocity'], n_bins, density=True, color=[5/255, 190/255, 120/255])
ax4.hist(axo_data['tail beat velocity'], n_bins, density=True, color=[242/255, 183/255, 124/255])
plt.xlabel('Tail velocity (°)')
plt.ylabel('Probability density')
p_tailvel, d_tailvel = ks_test(xen_data['tail beat velocity'], axo_data['tail beat velocity'], 2)

xen_freqs = xen_data['instantaneous half beat tail frequency'][xen_data['instantaneous half beat tail frequency']>=1]
axo_freqs = axo_data['instantaneous half beat tail frequency'][axo_data['instantaneous half beat tail frequency']>=1]
f5, ax5 = plt.subplots()
n_bins = np.arange(0,30,1)
ax5.hist(xen_freqs, n_bins, density=True, color=[5/255, 190/255, 120/255], alpha=0.7)
ax5.hist(axo_freqs, n_bins, density=True, color=[242/255, 183/255, 124/255], alpha=0.7)
plt.xlabel('Instantaneous half-beat tail frequency (°)')
plt.ylabel('# of detected tail beats')
p_freqs, d_freqs = ks_test(xen_freqs, axo_freqs)

f6, ax6 = plt.subplots()
n_bins = np.arange(0,1000,10)
ax6.hist(xen_data['head angular acceleration'], n_bins, density=True, color=[5/255, 190/255, 120/255])
ax6.hist(axo_data['head angular acceleration'], n_bins, density=True, color=[242/255, 183/255, 124/255])
plt.xlabel('Angular head acceleration (°/s/s)')
plt.ylabel('probability density')
p_accel, d_accel = ks_test(xen_data['head angular acceleration'], axo_data['head angular acceleration'], 1)

f7, ax7 = plt.subplots()
ax7.plot(abs(xen_data['head angular acceleration']), xen_data['instantaneous half beat tail frequency'], '.', color=[5/255, 190/255, 120/255])
ax7.plot(abs(axo_data['head angular acceleration']), axo_data['instantaneous half beat tail frequency'], '.', color=[242/255, 183/255, 124/255])
s_xen, rsq_xen = linear_regression_force_zero(xen_data['head angular acceleration'], xen_data['instantaneous half beat tail frequency'])