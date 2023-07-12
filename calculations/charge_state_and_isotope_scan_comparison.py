#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to compare bunch intensities from four different scenarios, with ion species of best charge state and isotope 
"""
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

# Load ion data and initialize for test for bunch intensities 
savefig = True
ion_data = pd.read_csv("../data/Ion_species.csv", sep=';', header=0, index_col=0).T
mass_number = ion_data.loc['A']
ion_type = 'Pb'

# Compare to reference intensities - WG5 and Roderik
ref_Table_SPS = pd.read_csv('../data/test_and_benchmark_data/SPS_final_intensities_WG5_and_Hannes.csv', delimiter=';', index_col=0)

#### PLOT SETTINGS #######
SMALL_SIZE = 8
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

def plot_Nb_dataframes(
                        df,
                        df2,
                        df3,
                        df4,
                        df_best_charge_state,
                        df_best_isotope,
                        output_extra_str = '',
                        ):

     
    # Define bar width for bar plot
    bar_width5 = 0.09
    x = np.arange(len(df.index))
    
    # Charge state - all scenarios, including LEIR-PS stripping and NO PS splitting
    fig5, ax5 = plt.subplots(1, 1, figsize = (6,5))
    bar51 = ax5.bar(x, ref_Table_SPS['WG5 Intensity'].astype(float)*mass_number, bar_width5, color='red', label='WG5') #
    bar52 = ax5.bar(x + bar_width5, df['LHC_ionsPerBunch'].astype(float)*mass_number, bar_width5, color='blue', label='Baseline scenario') #
    bar53 = ax5.bar(x + 2*bar_width5, df2['LHC_ionsPerBunch'].astype(float)*mass_number, bar_width5, color='gold', label='No PS splitting') #
    bar54 = ax5.bar(x + 3*bar_width5, df3['LHC_ionsPerBunch'].astype(float)*mass_number, bar_width5, color='limegreen', label='LEIR-PS stripping') #
    bar55 = ax5.bar(x + 4*bar_width5, df4['LHC_ionsPerBunch'].astype(float)*mass_number, bar_width5, color='gray', label='LEIR-PS stripping, \nno PS splitting') #
    bar56 = ax5.bar(x + 5*bar_width5, df_best_charge_state['2_Nb_best']*mass_number, bar_width5, color='green', label='Best LEIR charge state,\nno PS splitting') #
    bar57 = ax5.bar(x + 6*bar_width5, df_best_charge_state['4_Nb_best']*mass_number, bar_width5, color='cyan', label='Best LEIR charge state,\nno LEIR-PS strip + PS splitting') #
    ax5.set_xticks(x + 2*bar_width5)
    ax5.set_xticklabels(df.index)
    ax5.set_ylabel("Nucleons per bunch")
    ax5.legend()
    fig5.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    if savefig:
        fig5.savefig('../output/figures/combined_isotope_and_charge_state_scan/best_charge_states{}.png'.format(output_extra_str), dpi=250)
    plt.close()

    # Isotope check - All scenarios, including LEIR-PS stripping and NO PS splitting - also include charge state and isotope scan 
    fig6, ax6 = plt.subplots(1, 1, figsize = (6,5))
    ax6.bar(x, ref_Table_SPS['WG5 Intensity'].astype(float)*mass_number, bar_width5, color='red', label='WG5') #
    ax6.bar(x + bar_width5, df['LHC_ionsPerBunch'].astype(float)*mass_number, bar_width5, color='blue', label='Baseline scenario') #
    ax6.bar(x + 2*bar_width5, df2['LHC_ionsPerBunch'].astype(float)*mass_number, bar_width5, color='gold', label='No PS splitting') #
    ax6.bar(x + 3*bar_width5, df3['LHC_ionsPerBunch'].astype(float)*mass_number, bar_width5, color='limegreen', label='LEIR-PS stripping') #
    ax6.bar(x + 4*bar_width5, df4['LHC_ionsPerBunch'].astype(float)*mass_number, bar_width5, color='gray', label='LEIR-PS stripping, \nno PS splitting') #
    ax6.bar(x + 5*bar_width5, df_best_isotope['2_Nb_best']*mass_number, bar_width5, color='green', label='Best isotope,\nno PS splitting') #
    ax6.bar(x + 6*bar_width5, df_best_isotope['4_Nb_best']*mass_number, bar_width5, color='cyan', label='Best isotope,\nno LEIR-PS strip + PS splitting') #
    ax6.set_xticks(x + 2*bar_width5)
    ax6.set_xticklabels(df.index)
    ax6.set_ylabel("Nucleons per bunch")
    ax6.legend()
    fig6.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    if savefig:
        fig6.savefig('../output/figures/combined_isotope_and_charge_state_scan/best_isotopes{}.png'.format(output_extra_str), dpi=250)
    plt.close()
    
    # Isotope and charge state check check - NO PS splitting - also include charge state and isotope scan 
    fig7, ax7 = plt.subplots(1, 1, figsize = (6,5))
    ax7.bar(x - bar_width5, ref_Table_SPS['WG5 Intensity'].astype(float)*mass_number, bar_width5, color='red', label='WG5') #
    ax7.bar(x , df['LHC_ionsPerBunch'].astype(float)*mass_number, bar_width5, color='blue', label='Baseline scenario') #
    ax7.bar(x + bar_width5, df2['LHC_ionsPerBunch'].astype(float)*mass_number, bar_width5, color='gold', label='No PS splitting') #
    ax7.bar(x + 2*bar_width5, df3['LHC_ionsPerBunch'].astype(float)*mass_number, bar_width5, color='limegreen', label='LEIR-PS stripping') #
    ax7.bar(x + 3*bar_width5, df4['LHC_ionsPerBunch'].astype(float)*mass_number, bar_width5, color='gray', label='LEIR-PS stripping, \nno PS splitting') #
    ax7.bar(x + 4*bar_width5, df_best_charge_state['2_Nb_best']*mass_number, bar_width5, color='green', label='Best LEIR charge state,\nno PS splitting') #
    ax7.bar(x + 5*bar_width5, df_best_isotope['2_Nb_best']*mass_number, bar_width5, color='cyan', label='Best isotope,\nno PS splitting') #
    ax7.set_xticks(x + 2*bar_width5)
    ax7.set_xticklabels(df.index)
    ax7.set_ylabel("Nucleons per bunch")
    ax7.legend()
    fig7.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    if savefig:
        fig7.savefig('../output/figures/combined_isotope_and_charge_state_scan/best_charge_states_and_isotopes{}.png'.format(output_extra_str), dpi=250)
    plt.close()


if __name__ == '__main__': 
    # Load data from previous runs, with and without space charge
    df1 = pd.read_csv('../output/csv_tables/1_baseline.csv', index_col=0).T
    df2 = pd.read_csv('../output/csv_tables/2_no_PS_splitting.csv', index_col=0).T
    df3 = pd.read_csv('../output/csv_tables/3_LEIR_PS_stripping.csv', index_col=0).T
    df4 = pd.read_csv('../output/csv_tables/4_no_PS_splitting_and_LEIR_PS_stripping.csv', index_col=0).T
    
    df1_ps_sc = pd.read_csv('../output/csv_tables/1_baseline_with_PS_space_charge_limit.csv', index_col=0).T
    df2_ps_sc = pd.read_csv('../output/csv_tables/2_no_PS_splitting_with_PS_space_charge_limit.csv', index_col=0).T
    df3_ps_sc = pd.read_csv('../output/csv_tables/3_LEIR_PS_stripping_with_PS_space_charge_limit.csv', index_col=0).T
    df4_ps_sc = pd.read_csv('../output/csv_tables/4_no_PS_splitting_and_LEIR_PS_stripping_with_PS_space_charge_limit.csv', index_col=0).T
    
    # Data from charge state scan 
    df_best_charge_state = pd.read_csv('../output/csv_tables/charge_state_scan/best_charge_state/best_Nb_leir_charge_state_scan.csv', index_col=0)
    df_best_charge_state_ps_sc = pd.read_csv('../output/csv_tables/charge_state_scan/best_charge_state/best_Nb_leir_charge_state_scan_with_PC_SC_limit.csv', index_col=0)
    
    # Data from isotope scan
    df_best_isotope = pd.read_csv('../output/csv_tables/isotope_scan/best_isotope/best_Nb_isotope_scan.csv', index_col=0)
    df_best_isotope_ps_sc = pd.read_csv('../output/csv_tables/isotope_scan/best_isotope/best_Nb_isotope_scan_with_PC_SC_limit.csv', index_col=0)
    
    # First check case without PS space charge 
    plot_Nb_dataframes(
                        df1,
                        df2,
                        df3,
                        df4,
                        df_best_charge_state,
                        df_best_isotope
                        )
    
    # Then check case with PS space charge 
    plot_Nb_dataframes(
                        df1_ps_sc,
                        df2_ps_sc,
                        df3_ps_sc,
                        df4_ps_sc,
                        df_best_charge_state_ps_sc,
                        df_best_isotope_ps_sc,
                        output_extra_str='__with_PS_SC_limit'
                        )