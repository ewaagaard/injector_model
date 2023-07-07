#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script to calculate final nucleon intensity into the LHC considering linear space charge limit in LEIR and SPS,
with linear space charge limit in the PS 
"""
import matplotlib.pyplot as plt
import pandas as pd 
from injector_model import InjectorChain
import numpy as np

# Load ion data and initialize for test for bunch intensities 
ion_data = pd.read_csv("../data/Ion_species.csv", sep=';', header=0, index_col=0).T
ion_type = 'Pb'

# Compare to reference intensities
ref_Table_SPS = pd.read_csv('../data/SPS_final_intensities_WG5_and_Hannes.csv', delimiter=';', index_col=0)

# Calculate the bunch intensity going into the LHC - now Roderik accounts for SPS transmission
# Roderik uses Reyes excel as input table for linac3: 200 us pulse length, 70 uA
injector_chain2 = InjectorChain(ion_type, 
                                ion_data, 
                                nPulsesLEIR = 0,
                                LEIR_bunches = 2,
                                PS_splitting = 2,
                                account_for_SPS_transmission=True,
                                consider_PS_space_charge_limit=True
                                )
result = injector_chain2.calculate_LHC_bunch_intensity()

# Calculate LHC bunch intensity for all ions
output_1 = '1_baseline_with_PS_space_charge_limit'
df = injector_chain2.calculate_LHC_bunch_intensity_all_ion_species(save_csv=True, output_name = output_1)


## TRY WITHOUT PS SPLITTING
output_2 ='2_no_PS_splitting_with_PS_space_charge_limit'
injector_chain3 = InjectorChain(ion_type, 
                                ion_data, 
                                nPulsesLEIR = 0,
                                LEIR_bunches = 2,
                                PS_splitting = 1,
                                account_for_SPS_transmission=True,
                                consider_PS_space_charge_limit=True
                                )
df3 = injector_chain3.calculate_LHC_bunch_intensity_all_ion_species(save_csv=True, output_name=output_2)


## WITH PS SPLITTING AND LEIR-PS STRIPPING
output_3 = '3_LEIR_PS_stripping_with_PS_space_charge_limit'
injector_chain4 = InjectorChain(ion_type, 
                                ion_data, 
                                nPulsesLEIR = 0,
                                LEIR_bunches = 2,
                                PS_splitting = 2,
                                account_for_SPS_transmission=True,
                                LEIR_PS_strip=True,
                                consider_PS_space_charge_limit=True
                                )
df4 = injector_chain4.calculate_LHC_bunch_intensity_all_ion_species(save_csv=True, output_name=output_3)

## WITH NO SPLITTING AND LEIR-PS STRIPPING
output_4 = '4_no_PS_splitting_and_LEIR_PS_stripping_with_PS_space_charge_limit'
injector_chain5 = InjectorChain(ion_type, 
                                ion_data, 
                                nPulsesLEIR = 0,
                                LEIR_bunches = 2,
                                PS_splitting = 1,
                                account_for_SPS_transmission=True,
                                LEIR_PS_strip=True,
                                consider_PS_space_charge_limit=True
                                )
df5 = injector_chain5.calculate_LHC_bunch_intensity_all_ion_species(save_csv=True, output_name=output_4)

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

# Define bar width for bar plot
bar_width = 0.35
x = np.arange(len(df.index))

fig, ax = plt.subplots(1, 1, figsize = (6,5))
fig.suptitle("")
bar2 = ax.bar(x - bar_width/2, ref_Table_SPS['WG5 Intensity']*df['atomicNumber'], bar_width, color='red', label='WG5') #
bar1 = ax.bar(x + bar_width/2, df['LHC_chargesPerBunch'], bar_width, color='blue', label='Baseline scenario') #
ax.set_xticks(x)
ax.set_xticklabels(df.index)
ax.set_ylabel("LHC charges per bunch")
ax.legend()
fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig.savefig('../output/{}_ChargesPerBunch.png'.format(output_1), dpi=250)

# Baseline scenario
fig2, ax2 = plt.subplots(1, 1, figsize = (6,5))
bar22 = ax2.bar(x - bar_width/2, ref_Table_SPS['WG5 Intensity']*df['massNumber'], bar_width, color='red', label='WG5') #
bar12 = ax2.bar(x + bar_width/2, df['LHC_ionsPerBunch']*df['massNumber'], bar_width, color='blue', label='Baseline scenario') #
ax2.set_xticks(x)
ax2.set_xticklabels(df.index)
ax2.set_ylabel("Nucleons per bunch")
ax2.legend()
fig2.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig2.savefig('../output/{}_NucleonsPerBunch.png'.format(output_1), dpi=250)

# No PS splitting 
bar_width2 = 0.25
fig3, ax3 = plt.subplots(1, 1, figsize = (6,5))
bar31 = ax3.bar(x - bar_width2, ref_Table_SPS['WG5 Intensity']*df['massNumber'], bar_width2, color='red', label='WG5') #
bar32 = ax3.bar(x, df['LHC_ionsPerBunch']*df['massNumber'], bar_width2, color='blue', label='Baseline scenario') #
bar33 = ax3.bar(x + bar_width2, df3['LHC_ionsPerBunch']*df3['massNumber'], bar_width2, color='gold', label='No PS splitting') #
ax3.set_xticks(x)
ax3.set_xticklabels(df.index)
ax3.set_ylabel("Nucleons per bunch")
ax3.legend()
fig3.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig3.savefig('../output/{}.png'.format(output_2), dpi=250)

# Interpretation - Ca and Xe higher intensity due to higher LEIR charge state 
# for In, LEIR is the limitation, i.e. can only inject intensities in the SPS that are below space charge limit

# LEIR-PS stripping
bar_width4 = 0.2
fig4, ax4 = plt.subplots(1, 1, figsize = (6,5))
bar41 = ax4.bar(x - 1.5*bar_width4, ref_Table_SPS['WG5 Intensity']*df['massNumber'], bar_width4, color='red', label='WG5') #
bar42 = ax4.bar(x - 0.5*bar_width4, df['LHC_ionsPerBunch']*df['massNumber'], bar_width4, color='blue', label='Baseline scenario') #
bar43 = ax4.bar(x + 0.5*bar_width4, df3['LHC_ionsPerBunch']*df3['massNumber'], bar_width4, color='gold', label='No PS splitting') #
bar44 = ax4.bar(x + 1.5*bar_width4, df4['LHC_ionsPerBunch']*df4['massNumber'], bar_width4, color='limegreen', label='LEIR-PS stripping') #
ax4.set_xticks(x)
ax4.set_xticklabels(df.index)
ax4.set_ylabel("Nucleons per bunch")
ax4.legend()
fig4.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig4.savefig('../output/{}.png'.format(output_3), dpi=250)

# LEIR-PS stripping and NO PS splitting
bar_width5 = 0.15
fig5, ax5 = plt.subplots(1, 1, figsize = (6,5))
bar51 = ax5.bar(x, ref_Table_SPS['WG5 Intensity']*df['massNumber'], bar_width5, color='red', label='WG5') #
bar52 = ax5.bar(x + bar_width5, df['LHC_ionsPerBunch']*df['massNumber'], bar_width5, color='blue', label='Baseline scenario') #
bar53 = ax5.bar(x + 2*bar_width5, df3['LHC_ionsPerBunch']*df3['massNumber'], bar_width5, color='gold', label='No PS splitting') #
bar54 = ax5.bar(x + 3*bar_width5, df4['LHC_ionsPerBunch']*df4['massNumber'], bar_width5, color='limegreen', label='LEIR-PS stripping') #
bar55 = ax5.bar(x + 4*bar_width5, df5['LHC_ionsPerBunch']*df5['massNumber'], bar_width5, color='gray', label='LEIR-PS stripping, \nno PS splitting') #
ax5.set_xticks(x + 2*bar_width5)
ax5.set_xticklabels(df.index)
ax5.set_ylabel("Nucleons per bunch")
ax5.legend()
fig5.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig5.savefig('../output/{}.png'.format(output_4), dpi=250)

# A lower charge state in LEIR pushes the LEIR space charge limit further
# Fully stripped ions in PS and SPS means higher SPS space charge limit 
# O and Ar are almost at WG5 intensity, but Xe suffers from the stripping 