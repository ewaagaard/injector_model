#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script to calculate final nucleon intensity into the LHC considering linear space charge limit in LEIR, PS and SPS
- relativistic gamma for injection energies is read from data provided by injection_energies module
- also increase Brho in LEIR to 6.7 Tm
"""
import matplotlib.pyplot as plt
import pandas as pd 
from injector_model import InjectorChain
import numpy as np

# Load ion data and initialize for test for bunch intensities 
ion_data = pd.read_csv("../data/Ion_species.csv", sep=';', header=0, index_col=0).T
ion_type = 'Pb'

# Compare to reference intensities
ref_Table_SPS = pd.read_csv('../data/test_and_benchmark_data/SPS_final_intensities_WG5_and_Hannes.csv', delimiter=';', index_col=0)

# Calculate the bunch intensity going into the LHC - now Roderik accounts for SPS transmission
# Roderik uses Reyes excel as input table for linac3: 200 us pulse length, 70 uA
injector_chain2 = InjectorChain(ion_type, 
                                ion_data, 
                                nPulsesLEIR = 0,
                                LEIR_bunches = 2,
                                PS_splitting = 2,
                                account_for_SPS_transmission=True,
                                consider_PS_space_charge_limit=True,
                                use_gammas_ref=True
                                )
result = injector_chain2.calculate_LHC_bunch_intensity()

# Calculate LHC bunch intensity for all ions
print("\nCase 1:")
output_1 = '1_baseline_with_correct_gammas_and_PS_space_charge_limit'
df = injector_chain2.calculate_LHC_bunch_intensity_all_ion_species(save_csv=True, output_name = output_1)


## TRY WITHOUT PS SPLITTING
print("\nCase 2:")
output_2 ='2_no_PS_splitting_with_correct_gammas_and_PS_space_charge_limit'
injector_chain3 = InjectorChain(ion_type, 
                                ion_data, 
                                nPulsesLEIR = 0,
                                LEIR_bunches = 2,
                                PS_splitting = 1,
                                account_for_SPS_transmission=True,
                                consider_PS_space_charge_limit=True,
                                use_gammas_ref=True,
                                )
df3 = injector_chain3.calculate_LHC_bunch_intensity_all_ion_species(save_csv=True, output_name=output_2)


## WITH PS SPLITTING AND LEIR-PS STRIPPING
print("\nCase 3:")
output_3 = '3_LEIR_PS_stripping_with_correct_gammas_and_PS_space_charge_limit'
injector_chain4 = InjectorChain(ion_type, 
                                ion_data, 
                                nPulsesLEIR = 0,
                                LEIR_bunches = 2,
                                PS_splitting = 2,
                                account_for_SPS_transmission=True,
                                LEIR_PS_strip=True,
                                consider_PS_space_charge_limit=True,
                                use_gammas_ref=True
                                )
df4 = injector_chain4.calculate_LHC_bunch_intensity_all_ion_species(save_csv=True, output_name=output_3)

## WITH NO SPLITTING AND LEIR-PS STRIPPING
print("\nCase 4:")
output_4 = '4_no_PS_splitting_and_LEIR_PS_stripping_with_correct_gammas_and_PS_space_charge_limit'
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

### INCREASE Brho to LEIR 
print("\nCase 5:")
output_L = '5_baseline_with_correct_gammas_and_PS_space_charge_limit_and_higher_LEIR_brho'
injector_chain_L = InjectorChain(ion_type, 
                                ion_data, 
                                nPulsesLEIR = 0,
                                LEIR_bunches = 2,
                                PS_splitting = 2,
                                account_for_SPS_transmission=True,
                                consider_PS_space_charge_limit=True,
                                use_gammas_ref=True,
                                higher_brho_LEIR=True
                                )
result_L = injector_chain_L.calculate_LHC_bunch_intensity()
df_L = injector_chain_L.calculate_LHC_bunch_intensity_all_ion_species(save_csv=True, output_name = output_L)


### INCREASE Brho to LEIR and do stripping
print("\nCase 6:")
output_L_strip = '6_baseline_with_correct_gammas_and_PS_space_charge_limit_and_higher_LEIR_brho_and_strip'
injector_chain_L_strip = InjectorChain(ion_type, 
                                ion_data, 
                                nPulsesLEIR = 0,
                                LEIR_bunches = 2,
                                PS_splitting = 2,
                                LEIR_PS_strip=True,
                                account_for_SPS_transmission=True,
                                consider_PS_space_charge_limit=True,
                                use_gammas_ref=True,
                                higher_brho_LEIR=True
                                )
result_L_strip = injector_chain_L_strip.calculate_LHC_bunch_intensity()
df_L_strip = injector_chain_L_strip.calculate_LHC_bunch_intensity_all_ion_species(save_csv=True, output_name = output_L_strip)

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

# Baseline scenario with higher Brho
bar_width_L = 0.12
fig_L, ax_L = plt.subplots(1, 1, figsize = (6,5))
bar_L1 = ax_L.bar(x, ref_Table_SPS['WG5 Intensity']*df['massNumber'], bar_width_L, color='red', label='WG5') #
bar_L2 = ax_L.bar(x + bar_width_L, df['LHC_ionsPerBunch']*df['massNumber'], bar_width_L, color='blue', label='Baseline scenario') #
bar_L3 = ax_L.bar(x + 2*bar_width_L, df3['LHC_ionsPerBunch']*df3['massNumber'], bar_width_L, color='gold', label='No PS splitting') #
bar_L4 = ax_L.bar(x + 3*bar_width_L, df4['LHC_ionsPerBunch']*df4['massNumber'], bar_width_L, color='limegreen', label='LEIR-PS stripping') #
bar_L5 = ax_L.bar(x + 4*bar_width_L, df5['LHC_ionsPerBunch']*df5['massNumber'], bar_width_L, color='gray', label='LEIR-PS stripping, \nno PS splitting') #
bar_L6 = ax_L.bar(x + 5*bar_width_L, df_L['LHC_ionsPerBunch']*df_L['massNumber'], bar_width_L, color='purple', label='Baseline with higher LEIR Brho') 
ax_L.set_xticks(x + 2*bar_width_L)
ax_L.set_xticklabels(df.index)
ax_L.set_ylabel("Nucleons per bunch")
ax_L.legend()
fig_L.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig_L.savefig('../output/{}.png'.format(output_L), dpi=250)

# Baseline scenario with higher Brho and stripping
bar_width_L_strip = 0.1
fig_L_strip, ax_L_strip = plt.subplots(1, 1, figsize = (6,5))
bar_L1_strip = ax_L_strip.bar(x, ref_Table_SPS['WG5 Intensity']*df['massNumber'], bar_width_L_strip, color='red', label='WG5') #
bar_L2_strip = ax_L_strip.bar(x + bar_width_L_strip, df['LHC_ionsPerBunch']*df['massNumber'], bar_width_L_strip, color='blue', label='Baseline scenario') #
bar_L3_strip = ax_L_strip.bar(x + 2*bar_width_L_strip, df3['LHC_ionsPerBunch']*df3['massNumber'], bar_width_L_strip, color='gold', label='No PS splitting') #
bar_L4_strip = ax_L_strip.bar(x + 3*bar_width_L_strip, df4['LHC_ionsPerBunch']*df4['massNumber'], bar_width_L_strip, color='limegreen', label='LEIR-PS stripping') #
bar_L5_strip = ax_L_strip.bar(x + 4*bar_width_L_strip, df5['LHC_ionsPerBunch']*df5['massNumber'], bar_width_L_strip, color='gray', label='LEIR-PS stripping, \nno PS splitting') #
bar_L6_strip = ax_L_strip.bar(x + 5*bar_width_L_strip, df_L['LHC_ionsPerBunch']*df_L['massNumber'], bar_width_L_strip, color='purple', label='Baseline with higher LEIR Brho') 
bar_L7_strip = ax_L_strip.bar(x + 6*bar_width_L_strip, df_L_strip['LHC_ionsPerBunch']*df_L_strip['massNumber'], bar_width_L_strip, color='magenta', label='LEIR-PS strip with \nhigher LEIR Brho') 
ax_L_strip.set_xticks(x + 2*bar_width_L_strip)
ax_L_strip.set_xticklabels(df.index)
ax_L_strip.set_ylabel("Nucleons per bunch")
ax_L_strip.legend(fontsize=10)
fig_L_strip.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig_L_strip.savefig('../output/{}.png'.format(output_L_strip), dpi=250)

# A lower charge state in LEIR pushes the LEIR space charge limit further
# Fully stripped ions in PS and SPS means higher SPS space charge limit 
# O and Ar are almost at WG5 intensity, but Xe suffers from the stripping 