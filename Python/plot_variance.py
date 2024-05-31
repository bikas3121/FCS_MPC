
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""  Plot the  variance plots for the given methods 

@author: Bikash Adhikari
@date: 24.03.2024
@license: BSD 3-Clause
"""


# imports
import numpy as np
import matplotlib.pyplot as plt


def plot_variance(**kwargs):
    """Plots the bar plots for the given values 
        : key, value 
    """
    x = []
    y = []
    for key, value in kwargs.items() :
        x.append(key)
        y.append(value)

    fig, ax = plt.subplots()
    x_pos = np.arange(0, len(x),1) 
    bar_labels = x
    ax.bar(x,y, label=bar_labels)
    plt.xticks([i  for i in range(len(y))],x[0:len(x)])
    for i in range(len(x)):
        plt.text(x = x_pos[i]-0.1, y= y[i] + 0.015 , s= y[i], size = 10)
    ax.set_ylabel('Error Variance')
    ax.legend()
    plt.show()

