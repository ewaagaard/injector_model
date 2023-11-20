#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to vary charge state in LEIR to find optimum state for maximum injected LHC intensity 
"""
import matplotlib.pyplot as plt
import pandas as pd 
from injector_model import InjectorChain
import numpy as np

#### PLOTTING PARAMETERS #######
SMALL_SIZE = 11
MEDIUM_SIZE = 17
BIGGER_SIZE = 23
plt.rcParams["font.family"] = "serif"
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)   # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
colors = ['green', 'blue', 'purple', 'brown', 'teal', 'coral', 'cyan', 'darkred']

# Load ion data and initialize for test for bunch intensities 
ion_data = pd.read_csv("../data/Ion_species.csv", header=0, index_col=0).T

# Set upper limit to Q / A for LINAC3 - LEIR transfer line
LinacLEIRlim = 0.25

# Compare to reference intensities - WG5 and Roderik
ref_Table_SPS = pd.read_csv('../data/test_and_benchmark_data/SPS_final_intensities_WG5_and_Hannes.csv', delimiter=';', index_col=0)
WG5_intensity = ref_Table_SPS['WG5 Intensity']
Roderik_LHC_charges_per_bunch = pd.read_csv('../data/test_and_benchmark_data/Roderik_2021_LHC_charges_per_bunch_output.csv', index_col=0)
ref_val = Roderik_LHC_charges_per_bunch.sort_values('Z')

# Define all relevant scenarios (baseline, stripping, PS splitting, etc) in a function
def calculate_LHC_intensities_all_scenarios_vary_charge_state(
                                            Q,
                                            ion_type, 
                                            consider_PS_space_charge_limit,
                                            use_gammas_ref
                                            ):
    
    ## CASE 1: BASELINE (default Pb production)
    injector_chain1 = InjectorChain(ion_type, 
                                    ion_data, 
                                    nPulsesLEIR = 0,
                                    LEIR_bunches = 2,
                                    PS_splitting = 2,
                                    consider_PS_space_charge_limit = consider_PS_space_charge_limit,
                                    use_gammas_ref = use_gammas_ref
                                    )
    injector_chain1.Q = Q  # update charge state before stripping
    result1 = injector_chain1.calculate_LHC_bunch_intensity()
    SC_limits1 = injector_chain1.space_charge_limit_effect_on_LHC_bunch_intensity()

    ## 2: TRY WITHOUT PS SPLITTING
    injector_chain2 = InjectorChain(ion_type, 
                                    ion_data, 
                                    nPulsesLEIR = 0,
                                    LEIR_bunches = 2,
                                    PS_splitting = 1,
                                    consider_PS_space_charge_limit = consider_PS_space_charge_limit,
                                    use_gammas_ref = use_gammas_ref
                                    )
    injector_chain2.Q = Q  # update charge state before stripping
    result2 = injector_chain2.calculate_LHC_bunch_intensity()
    SC_limits2 = injector_chain2.space_charge_limit_effect_on_LHC_bunch_intensity()
    
     
    ## 3: WITH PS SPLITTING AND LEIR-PS STRIPPING
    injector_chain3 = InjectorChain(ion_type, 
                                    ion_data, 
                                    nPulsesLEIR = 0,
                                    LEIR_bunches = 2,
                                    PS_splitting = 2,
                                    account_for_SPS_transmission=True,
                                    LEIR_PS_strip=True,
                                    consider_PS_space_charge_limit = consider_PS_space_charge_limit,
                                    use_gammas_ref = use_gammas_ref
                                    )
    injector_chain3.Q = Q  # update charge state before stripping
    result3 = injector_chain3.calculate_LHC_bunch_intensity()
    SC_limits3 = injector_chain3.space_charge_limit_effect_on_LHC_bunch_intensity()
    
    
    ## 4: WITH NO SPLITTING AND LEIR-PS STRIPPING
    injector_chain4 = InjectorChain(ion_type, 
                                    ion_data, 
                                    nPulsesLEIR = 0,
                                    LEIR_bunches = 2,
                                    PS_splitting = 1,
                                    account_for_SPS_transmission=True,
                                    LEIR_PS_strip=True,
                                    consider_PS_space_charge_limit = consider_PS_space_charge_limit,
                                    use_gammas_ref = use_gammas_ref
                                    )
    injector_chain4.Q = Q  # update charge state before stripping
    result4 = injector_chain4.calculate_LHC_bunch_intensity()
    SC_limits4 = injector_chain4.space_charge_limit_effect_on_LHC_bunch_intensity()
    
    return result1, result2, result3, result4, SC_limits1, SC_limits2, SC_limits3, SC_limits4

# Function to vary charge state and return a dictionary 
def vary_charge_state_and_plot(
                                output_name,
                                consider_PS_space_charge_limit, 
                                use_gammas_ref,
                                save_fig=True
                                ):
    
    # Empty array to contain all dataframes
    ion_dataframes = []
    
    # Ion figure
    num_rows = len(ion_data.T.index) // 2  # Integer division to determine the number of rows
    num_cols = 2  # Two columns
    
    # Create the combined figure with subplots
    fig0, axs = plt.subplots(num_rows, num_cols, figsize=(8.27, 10.2))
    #fig0.suptitle("Charge State Scan", fontsize=20)
    
    # Iterate over all ion species 
    count = 1
    for ion, row in ion_data.T.iterrows():
        
        print('\nVarying charge state for {}'.format(ion))
        
        Q_default = row['Q before stripping']
        Q_states = np.arange(1, row['Z']+1) # create array with all charge states between 1 and fully stripped ion
        
        # Create empty array for all the LHC bunch intensities, LEIR SC limit and SPS SC limit  
        Nb1_array, Nb2_array, Nb3_array, Nb4_array = np.zeros(len(Q_states)), np.zeros(len(Q_states)), np.zeros(len(Q_states)), np.zeros(len(Q_states)) 
        SC_SPS1_array, SC_SPS2_array, SC_SPS3_array, SC_SPS4_array = np.zeros(len(Q_states)), np.zeros(len(Q_states)), np.zeros(len(Q_states)), np.zeros(len(Q_states)) 
        SC_PS1_array, SC_PS2_array, SC_PS3_array, SC_PS4_array = np.zeros(len(Q_states)), np.zeros(len(Q_states)), np.zeros(len(Q_states)), np.zeros(len(Q_states))
        SC_LEIR1_array, SC_LEIR2_array, SC_LEIR3_array, SC_LEIR4_array = np.zeros(len(Q_states)), np.zeros(len(Q_states)), np.zeros(len(Q_states)), np.zeros(len(Q_states))
        gammas_SPS1_array, gammas_SPS2_array, gammas_SPS3_array, gammas_SPS4_array = np.zeros(len(Q_states)), np.zeros(len(Q_states)), np.zeros(len(Q_states)), np.zeros(len(Q_states)) 
        
        # First check default intensity from standard charge state
        result01, result02, result03, result04, SC_limits01, SC_limits02, SC_limits03, SC_limits04 = calculate_LHC_intensities_all_scenarios_vary_charge_state(
                                                                                                        Q_default,
                                                                                                        ion, 
                                                                                                        consider_PS_space_charge_limit,
                                                                                                        use_gammas_ref,
                                                                                                        )
        Nb0 = result01['LHC_ionsPerBunch'] # LHC bunch intensity for default baseline scenario and charge state 
        
        # Iterate over all the Q_states 
        for j, Q in enumerate(Q_states):
            result1, result2, result3, result4, SC_limits1, SC_limits2, SC_limits3, SC_limits4 = calculate_LHC_intensities_all_scenarios_vary_charge_state(
                                                                                                        Q,
                                                                                                        ion, 
                                                                                                        consider_PS_space_charge_limit,
                                                                                                        use_gammas_ref,
                                                                                                        )
            # Append bunch intensities, LEIR and SPS space charge limit
            Nb1_array[j] = result1['LHC_ionsPerBunch']
            Nb2_array[j] = result2['LHC_ionsPerBunch']
            Nb3_array[j] = result3['LHC_ionsPerBunch']
            Nb4_array[j] = result4['LHC_ionsPerBunch']
                        
            SC_LEIR1_array[j] = SC_limits1[0] 
            SC_LEIR2_array[j] = SC_limits2[0] 
            SC_LEIR3_array[j] = SC_limits3[0] 
            SC_LEIR4_array[j] = SC_limits4[0] 
            
            SC_PS1_array[j] = SC_limits1[1]
            SC_PS2_array[j] = SC_limits2[1]
            SC_PS3_array[j] = SC_limits3[1]
            SC_PS4_array[j] = SC_limits4[1]
            
            SC_SPS1_array[j] = SC_limits1[2]
            SC_SPS2_array[j] = SC_limits2[2]
            SC_SPS3_array[j] = SC_limits3[2]
            SC_SPS4_array[j] = SC_limits4[2]
            
            gammas_SPS1_array[j] = result1['SPS_gamma_inj']
            gammas_SPS2_array[j] = result2['SPS_gamma_inj']
            gammas_SPS3_array[j] = result3['SPS_gamma_inj']
            gammas_SPS4_array[j] = result4['SPS_gamma_inj']
            
        # Make dataframe and save
        dict_ion = {
                'Q_state': Q_states,
                'Nb0_1_Baseline': Nb1_array, 
                'Nb0_2_No_PS_split': Nb2_array,
                'Nb0_3_LEIR_PS_strip': Nb3_array, 
                'Nb0_4_LEIR_PS_strip_and_no_PS_split': Nb4_array,
                'LEIR_SC_limit_1_Baseline': SC_LEIR1_array, 
                'LEIR_SC_limit_2_No_PS_split': SC_LEIR2_array,
                'LEIR_SC_limit_3_LEIR_PS_strip': SC_LEIR3_array, 
                'LEIR_SC_limit_4_LEIR_PS_strip_and_no_PS_split': SC_LEIR4_array,
                'PS_SC_limit_1_Baseline': SC_PS1_array, 
                'PS_SC_limit_2_No_PS_split': SC_PS2_array,
                'PS_SC_limit_3_LEIR_PS_strip': SC_PS3_array, 
                'PS_SC_limit_4_LEIR_PS_strip_and_no_PS_split': SC_PS4_array,
                'SPS_SC_limit_1_Baseline': SC_SPS1_array, 
                'SPS_SC_limit_2_No_PS_split': SC_SPS2_array,
                'SPS_SC_limit_3_LEIR_PS_strip': SC_SPS3_array, 
                'SPS_SC_limit_4_LEIR_PS_strip_and_no_PS_split': SC_SPS4_array,
                'SPS_gamma_inj_1_Baseline': gammas_SPS1_array, 
                'SPS_gamma_inj_2_No_PS_split': gammas_SPS2_array,
                'SPS_gamma_inj_3_LEIR_PS_strip': gammas_SPS3_array, 
                'SPS_gamma_inj_4_LEIR_PS_strip_and_no_PS_split': gammas_SPS4_array,
                 }
        df = pd.DataFrame(dict_ion)
        df = df.set_index(['Q_state'])
        if save_fig:
            df.to_csv('output/csv_tables/charge_state_scan/{}_leir_charge_state_scan{}.csv'.format(ion, output_name))
        ion_dataframes.append(df)
        
        #### PLOTTING - Make figure for all the charge states ####
        fig, ax = plt.subplots(1, 1, figsize = (6,5))
        #fig.suptitle(ion, fontsize=20)
        if row['Z'] > 2.0:
            ax.axvspan(0.0, np.max(Q_states[Q_states / row['A'] < LinacLEIRlim]), alpha=0.25, color='coral', label='Not accessible LINAC3-LEIR')
        ax.plot(Q_states, Nb1_array, color='blue', linewidth=3, linestyle='-', label='1: Baseline')
        ax.plot(Q_states, Nb2_array, linestyle='--', color='gold', linewidth=3, label='2: No PS splitting') #
        ax.plot(Q_states, Nb3_array, linestyle='-.', color='limegreen', linewidth=3, label='3: LEIR-PS stripping') #
        ax.plot(Q_states, Nb4_array, linestyle='--', color='gray', linewidth=3, label='4: LEIR-PS stripping, \nno PS splitting') #
        ax.plot(Q_default, Nb0, 'ro', markersize=13, alpha=0.8, label='1: Baseline with default charge state')
        if WG5_intensity[ion] > 0.0:
            ax.axhline(y = WG5_intensity[ion], color='red', label='WG5')
        ax.set_ylabel('LHC bunch intensity')
        ax.set_xlabel('LEIR charge state')
        ax.legend(fontsize=9)
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        if save_fig:
            fig.savefig('output/figures/charge_state_scan/{}_{}_leir_charge_state_scan{}.png'.format(count, ion, output_name), dpi=250)
        plt.close()
        
        #### PLOTTING SC limits ##### - Make figure for the LEIR and SPS space charge limits 
        
        ## 1) baseline case ####
        fig2, ax2 = plt.subplots(1, 1, figsize = (6,5))
        #fig2.suptitle(ion, fontsize=20)
        if row['Z'] > 2.0:
            ax2.axvspan(0.0, np.max(Q_states[Q_states / row['A'] < LinacLEIRlim]), alpha=0.25, color='coral', label='Not accessible LINAC3-LEIR')
        ax2.plot(Q_states, SC_LEIR1_array, color='brown', linewidth=4, linestyle='-', label='LEIR SC limit: Baseline')
        ax2.plot(Q_states, SC_PS1_array, color='slategrey', linewidth=4, linestyle=':', label='PS SC limit: Baseline')
        ax2.plot(Q_states, SC_SPS1_array, color='royalblue', linewidth=4, linestyle='--', label='SPS SC limit: Baseline')
        ax2.plot(Q_default, Nb0, 'ro', markersize=12, alpha=0.8, label='Baseline with default charge state')
        ax2.set_ylabel('LHC bunch intensity')
        ax2.set_xlabel('LEIR charge state')
        ax2.legend()
        ax2.set_ylim(0, 2 * np.max(SC_SPS1_array))
        fig2.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        if save_fig:
            fig2.savefig('output/figures/charge_state_scan/SC_limit/1_baseline_SC_limit_{}_leir_charge_state_scan{}.png'.format(ion, output_name), dpi=250)
        plt.close()
        
        ## 2) NO PS split ####
        fig3, ax3 = plt.subplots(1, 1, figsize = (6,5))
        if row['Z'] > 2.0:
            ax3.axvspan(0.0, np.max(Q_states[Q_states / row['A'] < LinacLEIRlim]), alpha=0.25, color='coral', label='Not accessible LINAC3-LEIR')
        ax3.plot(Q_states, SC_LEIR2_array, color='brown', linewidth=4, linestyle='-', label='LEIR SC limit: No PS split')
        ax3.plot(Q_states, SC_PS2_array, color='slategrey', linewidth=4, linestyle=':', label='PS SC limit: No PS split')
        ax3.plot(Q_states, SC_SPS2_array, color='royalblue', linewidth=4, linestyle='--', label='SPS SC limit: No PS split')
        ax3.plot(Q_default, Nb0, 'ro', markersize=12, alpha=0.8, label='Baseline with default charge state')
        ax3.set_ylabel('LHC bunch intensity')
        ax3.set_xlabel('LEIR charge state')
        ax3.legend()
        ax3.set_ylim(0, 2 * np.max(SC_SPS2_array))
        fig3.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        if save_fig:
            fig3.savefig('output/figures/charge_state_scan/SC_limit/2_NO_PS_split_SC_limit_{}_leir_charge_state_scan{}.png'.format(ion, output_name), dpi=250)
        plt.close()
        
        ## 3) LEIR-PS strip ####
        fig4, ax4 = plt.subplots(1, 1, figsize = (6,5))
        if row['Z'] > 2.0:
            ax4.axvspan(0.0, np.max(Q_states[Q_states / row['A'] < LinacLEIRlim]), alpha=0.25, color='coral', label='Not accessible LINAC3-LEIR')
        ax4.plot(Q_states, SC_LEIR3_array, color='brown', linewidth=4, linestyle='-', label='LEIR SC limit: LEIR-PS strip')
        ax4.plot(Q_states, SC_PS3_array, color='slategrey', linewidth=4, linestyle=':', label='PS SC limit: LEIR-PS strip')
        ax4.plot(Q_states, SC_SPS3_array, color='royalblue', linewidth=4, linestyle='--', label='SPS SC limit: LEIR-PS strip')
        ax4.plot(Q_default, Nb0, 'ro', markersize=12, alpha=0.8, label='Baseline with default charge state')
        ax4.set_ylabel('LHC bunch intensity')
        ax4.set_xlabel('LEIR charge state')
        ax4.legend()
        ax4.set_ylim(0, 2 * np.max(SC_SPS3_array))
        fig4.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        if save_fig:
            fig4.savefig('output/figures/charge_state_scan/SC_limit/3_LEIR_PS_strip_SC_limit_{}_leir_charge_state_scan{}.png'.format(ion, output_name), dpi=250)
        plt.close()
        
        ## 4) LEIR-PS strip ####
        fig5, ax5 = plt.subplots(1, 1, figsize = (6,5))
        if row['Z'] > 2.0:
            ax5.axvspan(0.0, np.max(Q_states[Q_states / row['A'] < LinacLEIRlim]), alpha=0.25, color='coral', label='Not accessible LINAC3-LEIR')
        ax5.plot(Q_states, SC_LEIR4_array, color='brown', linewidth=4, linestyle='-', label='LEIR SC limit: LEIR-PS strip\nand no PS split')
        ax5.plot(Q_states, SC_PS4_array, color='slategrey', linewidth=4, linestyle=':', label='PS SC limit: LEIR-PS strip\nand no PS split')
        ax5.plot(Q_states, SC_SPS4_array, color='royalblue', linewidth=4, linestyle='--', label='SPS SC limit: LEIR-PS strip\nand no PS split')
        ax5.plot(Q_default, Nb0, 'ro', markersize=12, alpha=0.8, label='Baseline with default charge state')
        ax5.set_ylabel('LHC bunch intensity')
        ax5.set_xlabel('LEIR charge state')
        ax5.legend()
        ax5.set_ylim(0, 2 * np.max(SC_SPS4_array))
        fig5.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        if save_fig:
            fig5.savefig('output/figures/charge_state_scan/SC_limit/4_LEIR_PS_strip_and_NO_PS_Split_SC_limit_{}_leir_charge_state_scan{}.png'.format(ion, output_name), dpi=250)
        plt.close()
        
        #########
                
        # Also fill in the big combined subplot
        row3 = (count-1) // num_cols  # Row index
        col3 = (count-1) % num_cols  # Column index
        ax3 = axs[row3, col3]  # Select the current subplot

        # Plot the data for the current ion
        if row['Z'] > 2.0:
            ax3.axvspan(0.0, np.max(Q_states[Q_states / row['A'] < LinacLEIRlim]), alpha=0.25, color='coral', label='Not accessible LINAC3-LEIR')
        if WG5_intensity[ion] > 0.0:
            ax3.axhline(y=WG5_intensity[ion], color='red', label='WG5')
        ax3.plot(Q_states, Nb1_array, color='blue', linewidth=3, linestyle='-', label='1: Baseline')
        ax3.plot(Q_states, Nb2_array, linestyle='--', color='gold', linewidth=3, label='2: No PS splitting')
        ax3.plot(Q_states, Nb3_array, linestyle='-.', color='limegreen', linewidth=3, label='3: LEIR-PS stripping')
        ax3.plot(Q_states, Nb4_array, linestyle='--', color='gray', linewidth=3, label='4: LEIR-PS stripping, \nno PS splitting')
        ax3.plot(Q_default, Nb0, 'ro', markersize=9, alpha=0.8, label='1: Baseline with default charge state')
        ax3.set_title(ion)  # Set the ion name as the title for the current subplot
        
        # Add legend in oxygen plot
        if count == 2:
            ax3.legend(fontsize=6)
        
        count += 1
    
    
    # Combined figure - Share y-axes for the same row
    # Share x-label for the same column
    for col in range(num_cols):
        axs[-1, col].set_xlabel('LEIR charge state', fontsize=13)
    
    # Share y-label for the same row
    for row in axs:
        row[0].set_ylabel('LHC bunch intensity', fontsize=13)
    
    #handles, labels = ax3.get_legend_handles_labels()
    #fig0.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0), fontsize=6)
    
    fig0.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)    
    # Save the combined figure
    if save_fig:
        fig0.savefig('output/figures/charge_state_scan/combined_leir_charge_state_scan{}.png'.format(output_name), dpi=250)
    plt.close()
    
    return ion_dataframes

if __name__ == '__main__': 
    
    # First check Roderik's case with no reference energy and PS limit  
    print('\nTesting without PS space charge limit... \n')
    dfs_0 = vary_charge_state_and_plot(
                                    output_name='',
                                    consider_PS_space_charge_limit=False, 
                                    use_gammas_ref=False,
                                    save_fig=True
                                    )


    # Then check case with reference gammas and PS space charge limit
    print('\nTesting with PS space charge limit... \n')
    dfs_1 = vary_charge_state_and_plot(
                                    output_name='_with_PS_SC_limit',
                                    consider_PS_space_charge_limit=True, 
                                    use_gammas_ref=False,
                                    save_fig=True 
                                    )
    
    # Store best charge state and injected LHC intensity 
    Q_best_1_array = np.zeros(len(ion_data.T.index))
    Q_best_2_array = np.zeros(len(ion_data.T.index))
    Q_best_3_array = np.zeros(len(ion_data.T.index))
    Q_best_4_array = np.zeros(len(ion_data.T.index))
    
    Nb_best_1_array = np.zeros(len(ion_data.T.index))
    Nb_best_2_array = np.zeros(len(ion_data.T.index))
    Nb_best_3_array = np.zeros(len(ion_data.T.index))
    Nb_best_4_array = np.zeros(len(ion_data.T.index))
    
    # Also considering PS space charge
    Q_best_1_array_ps_sc = np.zeros(len(ion_data.T.index))
    Q_best_2_array_ps_sc = np.zeros(len(ion_data.T.index))
    Q_best_3_array_ps_sc = np.zeros(len(ion_data.T.index))
    Q_best_4_array_ps_sc = np.zeros(len(ion_data.T.index))
    
    Nb_best_1_array_ps_sc = np.zeros(len(ion_data.T.index))
    Nb_best_2_array_ps_sc = np.zeros(len(ion_data.T.index))
    Nb_best_3_array_ps_sc = np.zeros(len(ion_data.T.index))
    Nb_best_4_array_ps_sc = np.zeros(len(ion_data.T.index))
    
    i = 0
    for ion, row in ion_data.T.iterrows():
        
        #### First check without PS space charge ####
        # Find the best ion species for LHC
        
        Q_best_1_array[i] = dfs_0[i]['Nb0_1_Baseline'].idxmax()
        Q_best_2_array[i] = dfs_0[i]['Nb0_2_No_PS_split'].idxmax()
        Q_best_3_array[i] = dfs_0[i]['Nb0_3_LEIR_PS_strip'].idxmax()
        Q_best_4_array[i] = dfs_0[i]['Nb0_4_LEIR_PS_strip_and_no_PS_split'].idxmax()
        
        Nb_best_1_array[i] = dfs_0[i]['Nb0_1_Baseline'].max()
        Nb_best_2_array[i] = dfs_0[i]['Nb0_2_No_PS_split'].max()
        Nb_best_3_array[i] = dfs_0[i]['Nb0_3_LEIR_PS_strip'].max()
        Nb_best_4_array[i] = dfs_0[i]['Nb0_4_LEIR_PS_strip_and_no_PS_split'].max()
    
        #### Then check PS space charge ####
        
        Q_best_1_array_ps_sc[i] = dfs_1[i]['Nb0_1_Baseline'].idxmax()
        Q_best_2_array_ps_sc[i] = dfs_1[i]['Nb0_2_No_PS_split'].idxmax()
        Q_best_3_array_ps_sc[i] = dfs_1[i]['Nb0_3_LEIR_PS_strip'].idxmax()
        Q_best_4_array_ps_sc[i] = dfs_1[i]['Nb0_4_LEIR_PS_strip_and_no_PS_split'].idxmax()
        
        Nb_best_1_array_ps_sc[i] = dfs_1[i]['Nb0_1_Baseline'].max()
        Nb_best_2_array_ps_sc[i] = dfs_1[i]['Nb0_2_No_PS_split'].max()
        Nb_best_3_array_ps_sc[i] = dfs_1[i]['Nb0_3_LEIR_PS_strip'].max()
        Nb_best_4_array_ps_sc[i] = dfs_1[i]['Nb0_4_LEIR_PS_strip_and_no_PS_split'].max()  
    
        i += 1
        
    # Create dictionaries with results 
    dict_best_ions = {'1_Q_best': Q_best_1_array,
                      '1_Nb_best': Nb_best_1_array,
                      '2_Q_best': Q_best_2_array,
                      '2_Nb_best': Nb_best_2_array,
                      '3_Q_best': Q_best_3_array,
                      '3_Nb_best': Nb_best_3_array,
                      '4_Q_best': Q_best_4_array,
                      '4_Nb_best': Nb_best_4_array,
                      }
    df_best_ions = pd.DataFrame(dict_best_ions)
    df_best_ions = df_best_ions.set_index(ion_data.T.index)
    df_best_ions.to_csv('output/csv_tables/charge_state_scan/best_charge_state/best_Nb_leir_charge_state_scan.csv', float_format='%e')
    
    dict_best_ions_ps_sc = {'1_Q_best': Q_best_1_array_ps_sc,
                      '1_Nb_best': Nb_best_1_array_ps_sc,
                      '2_Q_best': Q_best_2_array_ps_sc,
                      '2_Nb_best': Nb_best_2_array_ps_sc,
                      '3_Q_best': Q_best_3_array_ps_sc,
                      '3_Nb_best': Nb_best_3_array_ps_sc,
                      '4_Q_best': Q_best_4_array_ps_sc,
                      '4_Nb_best': Nb_best_4_array_ps_sc,
                      }
    df_best_ions_ps_sc = pd.DataFrame(dict_best_ions_ps_sc)
    df_best_ions_ps_sc = df_best_ions_ps_sc.set_index(ion_data.T.index)
    df_best_ions_ps_sc.to_csv('output/csv_tables/charge_state_scan/best_charge_state/best_Nb_leir_charge_state_scan_with_PC_SC_limit.csv', float_format='%e')
