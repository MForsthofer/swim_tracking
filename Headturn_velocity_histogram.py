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

Path = os.listdir()
for i in Path:
    if i.endswith('.xlsx'):
        excel_file = pd.ExcelFile(i)
        sheets = excel_file.sheet_names
        #velocities.append(pd.read_excel(excel_file, sheets[1]).iloc[:,1].values)
        arr = np.append(arr, pd.read_excel(excel_file, sheets[1]).iloc[:,1].values)
        n_bins = np.arange(0,1000,10)
        plt.hist(abs(arr), n_bins, color='blue')
        plt.xlim(10,1000)
        plt.ylim(0, 1000)
        plt.xlabel('Head turn velocity')
        plt.ylabel('count')