# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 19:32:18 2023

@author: mfors
"""
import pandas as pd
import os 
import matplotlib.pyplot as plt
import numpy as np

velocities = []
arr = np.array([])
vel_arr = np.zeros([0,1815])

Path = os.listdir()
for i in Path:
    if i.endswith('.xlsx'):
        excel_file = pd.ExcelFile(i)
        sheets = excel_file.sheet_names
        #velocities.append(pd.read_excel(excel_file, sheets[1]).iloc[:,1].values)
        arr = np.append(arr, np.diff(pd.read_excel(excel_file, sheets[1]).iloc[:,1].values))
        # ii = 0
        # while ii  <100:
        #     vel_arr = np.vstack([vel_arr, pd.read_excel(excel_file, sheets[1]).iloc[:,1].values[0:1815]])
        #     ii = ii+1
n_bins = np.arange(0,1000,10)
#xenopus
# plt.hist(abs(arr), n_bins, density=True, color=[5/255, 190/255, 120/255])
#axolotl
plt.hist(abs(arr), n_bins, density=True, color=[242/255, 183/255, 124/255])
plt.xlim(0,1000)
plt.ylim(0, 0.005)
plt.xlabel('Angular head acceleration (°)')
plt.ylabel('Probability density')

# f, ax = plt.subplots()
# im = ax.imshow(vel_arr)

