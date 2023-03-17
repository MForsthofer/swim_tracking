# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 13:19:29 2022

@author: forsthofer
"""


# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 10:47:54 2022

Script for loading in SLEAP tracking data

@author: forsthofer
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd
from scipy.fft import fft
from scipy.signal import savgol_filter
import json
from scipy.interpolate import interp1d
import os 

head_node = 2   # 1 for xenopus, 2 for axolotl
vel_node_1 = 0   #0 for xenopus, 0 for axolotl
vel_node_2 = 1   #3 for xenopus, 1 for axolotl
#filename = 'axolotl_tracking.000_A157_t1.analysis.h5'
pixel_distance = 0.31
framerate = 30 



def plot_pooled(pooled_domain, pooled_data, meancolor):
    '''plots list of pooled data as grey lines with an average'''
    f, ax = plt.subplots()
    data_stack = np.array(pooled_data).transpose()
    domain_stack = np.array(pooled_domain).transpose()
    mean_domain = np.mean(domain_stack, axis=1)
    ax.plot(domain_stack, data_stack, color='grey', linewidth=0.5)
    mean_fft = np.mean(data_stack,axis=1)
    ax.plot(mean_domain, mean_fft, color=meancolor, linewidth=1)
    ax.set_xlim(0.1, 15)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylim(0,90)
    ax.set_ylabel('Power')

def importtracking(filename):
    with h5py.File(filename, 'r') as f:
        occupancy_matrix = f['track_occupancy'][:]
        tracks_matrix = f['tracks'][:].T
    print(tracks_matrix.shape)
    return(occupancy_matrix, tracks_matrix)

def calc_swimming_distance(x_coordinates, y_coordinates, pixel_distance, framerate):
    """takes x and y coordinates of the tadpole position in pixels, and calculates the 
    distance swum by that animal. 
    pixel_distance: size of a pixel in mm, to calculate swimming distance in mm
    framerate: images per second by the camera, needed for swim speed calculation"""
    x_distance = np.diff(x_coordinates)*pixel_distance
    y_distance = np.diff(y_coordinates)*pixel_distance   
    swimming_distances = np.sqrt(np.square(x_distance)+np.square(y_distance))
    swimming_velocity = np.abs(np.diff(swimming_distances))*framerate
    return swimming_distances, swimming_velocity

def calc_turning(x_brain, y_brain, x_body, y_body, framerate):
    x_distance = (x_brain-x_body)*pixel_distance
    y_distance = (y_brain-y_body)*pixel_distance
    head_angles = np.rad2deg(np.arctan2(y_distance, x_distance))
    turn_velocity = np.diff(head_angles)
    for i in range(len(turn_velocity)):
        if turn_velocity[i]>300:
            turn_velocity[i] = 360-turn_velocity[i]
        elif turn_velocity[i]<-300:
            turn_velocity[i] = -360-turn_velocity[i]
    turn_velocity = turn_velocity*framerate
    return turn_velocity, head_angles

def animate_func(num, n_ax, xdata, ydata, xlim, ylim, xlabel, ylabel, title):
    """gradually plots a graph point by point, with the leading edge being a dot"""
    #wipe the graphic so it can be updated
    n_ax.clear()
    #plot the line
    n_ax.plot(xdata[:num+1], ydata[:num+1], c='red')
    
    #plot the leading point
    n_ax.plot(xdata[num],ydata[num], c='red', marker='o')
    #set axis limits
    n_ax.set_xlim(xlim)
    n_ax.set_ylim(ylim)
    n_ax.set_title(title)
    n_ax.set_xlabel(xlabel)
    n_ax.set_ylabel(ylabel)
    
def save_animation(anim, filename, foldername, framerate):
    f = foldername + filename
    writervideo = animation.FFMpegWriter(fps=framerate)
    anim.save(f, writer=writervideo)
    
def headturns_fft(data, time):
    #create fft of angular velocity frequency content
    N = len(data)
    T = np.mean(np.diff(time)) 
    fftdata = fft(data)
    tf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fftdata2 = np.abs(fftdata[0:N//2])
    freq_fft = 2.0/N * fftdata2
    plt.plot(tf, freq_fft, linewidth=1)
    plt.xlabel('frequency (Hz)')
    plt.ylabel('power (dB)')
    return (tf, freq_fft)

def fill_missing(Y, kind="linear"):
    
    """Fills missing values independently along each dimension after the first."""

    # Store initial shape.
    initial_shape = Y.shape

    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))

    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]

        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)
        
        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y

def pool_ffts(tag, label):  
    '''read and pool all fft json files in the folder'''
    path = os.listdir() 
    fft_stack = []   
    for i in path:
        if i.startswith(tag):
            with open(i, 'r') as openfile:
                # Reading from json file
                cur_file = json.load(openfile)
                fft_stack.append(cur_file[label])
    return fft_stack
    


def fft_manual_selection(time, data):
    '''Takes the swim distance traces, plots them, and lets you select a start point to make an fft. 
    following the start point it will indicated how big the analysis window will be and you can reselect. '''
    f, ax = plt.subplots()
    mngr = plt.get_current_fig_manager()
    # to put it into the upper left corner for example:
    ax.plot(data)
    ax.set_title('Swim distance')
    ax.set_ylabel('distance (mm)')
    ax.set_xlabel('data points')   
    #lets the user click in the plot. The x coordinate of the last click
    #will be the start of the part to analyze further, the second added dot the end. hit 
    #enter once you are happy with your selection. 
    x_y_click_coordinates = plt.ginput(-1, 0)
    plt.close()
    start_idx = int(round(x_y_click_coordinates[-2][0]))
    end_idx = int(round(x_y_click_coordinates[-1][0]))
    idx_window = end_idx-start_idx
    #cuts out the stimulus between your first and last click based on time. 
    swim_time = time[start_idx:end_idx]
    swim_segment = data[start_idx:end_idx]
    return (start_idx, end_idx, idx_window)

def freq_max(frequencies, spectrogram, lower_threshold):
    '''finds the maximum frequency in the spectrogram'''
    max_loc = np.argmax(spectrogram[lower_threshold:])
    max_freq = frequencies[max_loc+lower_threshold]
    return max_freq  

def resample(data, domain, resamp_domain):
    '''resamples time, stimulus and eye position signal to a set rate'''
    fdata = interp1d(domain, data,fill_value='extrapolate'); # get an interpolation function of the original stimulus data to the original time
    data_interp = fdata(resamp_domain) #apply the function to the resampled time
    return data_interp

    
Path = os.listdir()
for i in Path:
    if i.endswith('.h5'):
        filename = i
        
        #input video metadata

        
        
        #calculate results
        occupancy_matrix, tracks_matrix = importtracking(filename)
        tracks_matrix = fill_missing(tracks_matrix).T
        #tracks_matrix = savgol_filter(tracks_matrix, 7, 3)
        swimming_distance_thresh = 0.45
        
        swimming_distances, swimming_velocity = calc_swimming_distance(tracks_matrix[0,0,head_node,:], tracks_matrix[0,1,head_node,:],pixel_distance, framerate)
        swimming_distances[swimming_distances<swimming_distance_thresh] = 0
        swimming_velocity[swimming_distances[0:-1]<swimming_distance_thresh] = 0
        total_swimming_distance = np.max(np.cumsum(swimming_distances))
        cumulative_swimming_distance = np.cumsum(swimming_distances)
        
        angular_velocity, heading_direction = calc_turning(tracks_matrix[0,0,vel_node_1,:], tracks_matrix[0,1,vel_node_1,:], 
                                        tracks_matrix[0,0,vel_node_2,:], tracks_matrix[0,1,vel_node_2,:], framerate)
        moving_angular_velocity = np.nanmean(abs(angular_velocity[swimming_distances>swimming_distance_thresh]))
        
        t = 1/framerate * np.arange(1,len(angular_velocity)+1)
        
        #create a static plots
        f, ax = plt.subplots()
        ax.plot(tracks_matrix[0,0,head_node,:], tracks_matrix[0,1,head_node,:]*-1)
        mngr = plt.get_current_fig_manager()
        # to put it into the upper left corner for example:
        mngr.window.setGeometry(1,31,840, 840)
        ax.set_xlim(0,1024)
        ax.set_ylim(0,-1024)
        ax.set_xlabel('video width (px)')
        ax.set_ylabel('video height (px)')
        
        f2, ax2 = plt.subplots()
        ax2.plot(t[1:], swimming_velocity)
        mngr = plt.get_current_fig_manager()
        # mngr.window.setGeometry(1,31,420, 420)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Swim velocity (mm/s)')
        
        # f3, ax3 = plt.subplots()
        # ax3.plot(t, heading_direction+90)
        # mngr = plt.get_current_fig_manager()
        # # mngr.window.setGeometry(1,31,420, 420)
        # ax3.set_xlabel('Time (s)')
        # ax3.set_ylabel('Heading direction (°)')
        
        f1a, ax1a = plt.subplots(2,1)
        ax1a[0].plot(t, angular_velocity)
        ax1a[0].set_xlabel('Time (s)')
        ax1a[0].set_ylabel('Head velocity (°)')
        tf, freq_fft = headturns_fft(angular_velocity, t)
        ax1a[1].set_xlabel('Frequency')
        ax1a[1].set_ylabel('Power')
        
        fft_start, fft_end, active_fft_window = fft_manual_selection(t, cumulative_swimming_distance)
        f2a, ax2a = plt.subplots(2,1)
        ax2a[0].plot(t[fft_start:fft_end], angular_velocity[fft_start:fft_end])
        ax2a[0].set_xlabel('Time (s)')
        ax2a[0].set_ylabel('Head velocity (°)')
        tf_swim, freq_fft_swim = headturns_fft(angular_velocity[fft_start:fft_end], t[fft_start:fft_end])
        ax2a[1].set_xlabel('Frequency')
        ax2a[1].set_ylabel('Power')
        
        f3a, ax3a = plt.subplots(2,1)
        ax3a[0].plot(t, heading_direction[:-1]+90)
        mngr = plt.get_current_fig_manager()
        # mngr.window.setGeometry(1,31,420, 420)
        ax3a[0].set_xlabel('Time (s)')
        ax3a[0].set_ylabel('Heading direction (°)')
        #ax3a[1].plot(t, angular_velocity)
        direction_tf, direction_fft = headturns_fft(heading_direction, t)
        
        
            
        
        # ### create animated plots and save them
        # #save track of head turn velocity
        # f1, ax1 = plt.subplots()
        # mngr = plt.get_current_fig_manager()
        # # to put it into the upper left corner for example:
        # mngr.window.setGeometry(1,31,840, 420)
        # angular_velocity_animation = animation.FuncAnimation(f1, animate_func, fargs=(ax1, t, angular_velocity, [0, np.max(t)], [-2000, 2000], 'time (s)', 'angular head velocity (°/s)', 'angular head velocity', ), interval=1, frames=len(t))
        # save_animation(angular_velocity_animation, 'angular_velocity.mp4', r"D:\Sync&Share\projects\patty\xenopus\predictions", framerate)
        # print('saved angular video')
        
        
        
        # #save velocity plot
        # f3, ax3 = plt.subplots()
        # swim_velocity_animation = animation.FuncAnimation(f3, animate_func, fargs=(ax3, t[1:], swimming_velocity, [0, np.max(t[1:])], [np.min(swimming_velocity), np.max(swimming_velocity)], 
        #                                                                       'time (s)', 'swim velocity (mm/s)', 'swimming velocity', ), interval=1, frames=len(t[1:]))
        # save_animation(swim_velocity_animation, 'swim_velocity.mp4', r"D:\Sync&Share\projects\patty\xenopus\predictions", framerate)
        
        # #save distance plot
        # f4, ax4 = plt.subplots()
        # mngr = plt.get_current_fig_manager()
        # # to put it into the upper left corner for example:
        # mngr.window.setGeometry(1,31,840,420)
        # swim_distance_animation = animation.FuncAnimation(f4, animate_func, fargs=(ax4, t, cumulative_swimming_distance, [0, np.max(t)], [0, 3000],                                                                      'time (s)', 'swum distance (mm)', 'swimming distance', ), interval=1, frames=len(t[1:]))
        # save_animation(swim_distance_animation, 'swim_distance.mp4', r"D:\Sync&Share\projects\patty\xenopus\predictions", framerate)
        # print('saved distance video')
        
        # #save position plot
        # f2, ax2 = plt.subplots()
        # #plt.gca().set_aspect('equal', adjustable='box')
        # mngr = plt.get_current_fig_manager()
        # mngr.window.setGeometry(1,31,960, 536)
        # plt.pause(0.5)
        # position_animation = animation.FuncAnimation(f2, animate_func, fargs = (ax2, tracks_matrix[0,0,head_node,:], tracks_matrix[0,1,head_node,:]*-1, [0, 1920], [-1072, 0], 'video width (px)', 'video height (px)',                                                                        'animal position'), interval=1, frames=len(t))
        # save_animation(position_animation, 'position_track.mp4', r"D:\Sync&Share\projects\patty\xenopus\predictions", framerate)
        # print('saved position video')
        
        resamp_fft_swim = resample(freq_fft_swim, tf_swim, tf)
        direction_fft = resample(direction_fft, direction_tf, tf)
        
        fft_total_peak = freq_max(tf, freq_fft, 0)
        fft_swim_peak = freq_max(tf, resamp_fft_swim, 0)
        fft_direction_peak = freq_max(tf, direction_fft, 2)
        
        # fft_data = {'animal':filename[filename.rfind('.analysis')-7:filename.rfind('.analysis')-3] , 
        #             'repetition':filename[filename.rfind('.analysis')-2:filename.rfind('.analysis')],
        #             'freq_domain':list(tf),
        #             'fft':list(freq_fft),
        #             'swim_fft':list(freq_fft_swim),
        #             'swim_freq_domain':list(tf_swim)
        #             }
        
        # json_data = json.dumps(fft_data)
        
        results = pd.DataFrame(data=[total_swimming_distance], index=[filename[filename.rfind('.analysis')-7:filename.rfind('.analysis')-0]], columns=['swimming distance'])
        results['head_turn_velocity'] = moving_angular_velocity
        results['whole fft peak'] = fft_total_peak
        results['swimming fft peak'] = fft_swim_peak
        results['direction fft peak'] = fft_direction_peak
        results['fft_time'] = t[fft_end]-t[fft_start]
        results['fft_start'] = t[fft_start]
        
        
        
        # with open('fft_'+filename[filename.rfind('.analysis')-7:filename.rfind('.analysis')], "w") as outfile:
        #     outfile.write(json_data)
        
        # fft_stack = pool_ffts('fft_', 'fft')
        # tf_stack = pool_ffts('fft_', 'freq_domain')
        # swim_fft_stack = pool_ffts('fft_', 'swim_fft')
        # swim_tf_stack = pool_ffts('fft_', 'swim_freq_domain')
        # plot_pooled(tf_stack, fft_stack, 'blue')
        # plot_pooled(swim_tf_stack, swim_fft_stack, 'blue')
        ## save data to excel
        with pd.ExcelWriter(filename[filename.rfind('/')+1:filename.find('.h5')]+('tracking_results')+('.xlsx')) as writer:  
            results.T.to_excel(writer, sheet_name='total swimming distance')
            pd.DataFrame(angular_velocity).to_excel(writer, sheet_name='turn velocity')
            pd.DataFrame(freq_fft).to_excel(writer, sheet_name='total fft')
            pd.DataFrame(tf).to_excel(writer, sheet_name='frequency_domain')
            pd.DataFrame(resamp_fft_swim).to_excel(writer, sheet_name='swimming fft')
            pd.DataFrame(direction_fft).to_excel(writer, sheet_name='direction fft')
