#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 18:26:21 2023

@author: amadi
"""

import numpy as np

#%%

def time_conv(variable):
    
    """
    converts datesec variable to datetime format
    """
    
    wac_time = variable
    #print(wac_time_2)

    #convert the integer to HH:MM:SS format
    w_time = []

    for i in wac_time:

        time_int = i
        time = time_int/3600
        hours = int(time)
        minutes = (time*60) % 60
        seconds = (time*3600) % 60

        w_time = np.append(w_time, "%d:%02d:%02d" % (hours, minutes, seconds))

    return w_time

def time_2d(wtime):
    
    """
    slices the hour and minutes slice of datetime
    """
    
    wac_time_2d = []
    for i in wtime:
        waccy = i[0:5]
        wac_time_2d = np.append(wac_time_2d, waccy)

    return wac_time_2d