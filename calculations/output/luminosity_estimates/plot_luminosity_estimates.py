#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to plot 1-month integrated luminosity (AA) and nucleon-nucleon (NN) integrated luminosity 
from "conservative" (baseline) and "optimistic" (optimized charge state + no PS split) scenario
from CTE output from Roderik Bruce 
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#### PLOT THE DATA #######
SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 20
plt.rcParams["font.family"] = "serif"
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)   # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
colors = ['green', 'blue', 'purple', 'brown', 'teal', 'coral', 'cyan', 'darkred']


index = ['O', 'Ar', 'Ca', 'Kr', 'In', 'Xe', 'Pb']
x = np.arange(len(index))
bar_width = 0.25

df_AA = pd.read_csv('ion_AA_integrated_luminosity.csv', header=None)
df_NN = pd.read_csv('ion_NN_integrated_luminosity.csv', header=None)

# Plot integrated luminosity
fig, ax = plt.subplots(1, 1, figsize = (6,5))
ax.bar(x - bar_width, df_AA[0], bar_width, color='red', label='WG5') #
ax.bar(x, df_AA[1], bar_width, color='blue', label='Conservative') #
ax.bar(x + bar_width, df_AA[2], bar_width, color='green', label='Optimistic') #
ax.set_xticks(x)
ax.set_xticklabels(index)
ax.set_ylabel(r'$\int \mathcal{L}_{AA} dt$  [nb$^{-1}$]')
ax.legend()
ax.set_yscale('log')
fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig.savefig('AA_integrated_luminosity.png', dpi=250)

# Plot integrated nucleon-nucleon luminosity
fig2, ax2 = plt.subplots(1, 1, figsize = (6,5))
ax2.bar(x - bar_width, df_NN[0], bar_width, color='red', label='WG5') #
ax2.bar(x, df_NN[1], bar_width, color='blue', label='Conservative') #
ax2.bar(x + bar_width, df_NN[2], bar_width, color='green', label='Optimistic') #
ax2.set_xticks(x)
ax2.set_xticklabels(index)
ax2.set_ylabel(r'$\int \mathcal{L}_{NN} dt$  [pb$^{-1}$]')
ax2.legend()
fig2.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig2.savefig('NN_integrated_luminosity.png', dpi=250)