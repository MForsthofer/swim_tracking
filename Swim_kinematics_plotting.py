# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 19:32:18 2023

@author: mfors
"""
import pandas as pd
import os 
import matplotlib.pyplot as plt
import numpy as np
import pickle as pk

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

f4, ax4 = plt.subplots()
n_bins = np.arange(0,100,2)
ax4.hist(xen_data['tail beat velocity'], n_bins, density=True, color=[5/255, 190/255, 120/255])
ax4.hist(axo_data['tail beat velocity'], n_bins, density=True, color=[242/255, 183/255, 124/255])
plt.xlabel('Tail velocity (°)')
plt.ylabel('Probability density')

f5, ax5 = plt.subplots()
n_bins = np.arange(0,30,0.5)
ax5.hist(xen_data['instantaneous half beat tail frequency'], n_bins, density=True, color=[5/255, 190/255, 120/255])
ax5.hist(axo_data['instantaneous half beat tail frequency'], n_bins, density=True, color=[242/255, 183/255, 124/255])
plt.xlabel('Instantaneous tail frequency (°)')
plt.ylabel('# of detected tail beats')

f5, ax5 = plt.subplots()
n_bins = np.arange(0,30,0.5)
ax5.hist(xen_data['instantaneous half beat tail frequency'], n_bins, color=[5/255, 190/255, 120/255])
ax5.hist(axo_data['instantaneous half beat tail frequency'], n_bins, color=[242/255, 183/255, 124/255])
plt.xlabel('Instantaneous tail frequency (°)')
plt.ylabel('# of detected tail beats')

