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
SMALL_SIZE = 10
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

# Load ion data and initialize for test for bunch intensities 
ion_data = pd.read_csv("../data/Ion_species.csv", sep=';', header=0, index_col=0).T

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
    
    return result1, result2, result3, result4

# Function to vary charge state and return a dictionary 
def vary_charge_state_and_plot(
                                output_name,
                                consider_PS_space_charge_limit, 
                                use_gammas_ref,
                                save_fig=True
                                ):
    
    # Empty array to contain all dataframes
    ion_dataframes = []
    
    # Iterate over all ion species 
    count = 1
    for ion, row in ion_data.T.iterrows():
        
        print('\nVarying charge state for {}'.format(ion))
        
        Q_default = row['Q before stripping']
        Q_states = np.arange(1, row['Z']+1) # create array with all charge states between 1 and fully stripped ion
        
        # Create empty array for all the LHC bunch intensities, LEIR SC limit and SPS SC limit  
        Nb1_array, Nb2_array, Nb3_array, Nb4_array = np.zeros(len(Q_states)), np.zeros(len(Q_states)), np.zeros(len(Q_states)), np.zeros(len(Q_states)) 
        SC_SPS1_array, SC_SPS2_array, SC_SPS3_array, SC_SPS4_array = np.zeros(len(Q_states)), np.zeros(len(Q_states)), np.zeros(len(Q_states)), np.zeros(len(Q_states)) 
        SC_LEIR1_array, SC_LEIR2_array, SC_LEIR3_array, SC_LEIR4_array = np.zeros(len(Q_states)), np.zeros(len(Q_states)), np.zeros(len(Q_states)), np.zeros(len(Q_states))
        gammas_SPS1_array, gammas_SPS2_array, gammas_SPS3_array, gammas_SPS4_array = np.zeros(len(Q_states)), np.zeros(len(Q_states)), np.zeros(len(Q_states)), np.zeros(len(Q_states)) 
        
        # First check default intensity from standard charge state
        result01, result02, result03, result04 = calculate_LHC_intensities_all_scenarios_vary_charge_state(
                                                                                                        Q_default,
                                                                                                        ion, 
                                                                                                        consider_PS_space_charge_limit,
                                                                                                        use_gammas_ref,
                                                                                                        )
        Nb0 = result01['LHC_ionsPerBunch'] # LHC bunch intensity for default baseline scenario and charge state 
        
        # Iterate over all the Q_states 
        for j, Q in enumerate(Q_states):
            result1, result2, result3, result4 = calculate_LHC_intensities_all_scenarios_vary_charge_state(
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
                        
            SC_LEIR1_array[j] = result1['LEIR_space_charge_limit']
            SC_LEIR2_array[j] = result2['LEIR_space_charge_limit']
            SC_LEIR3_array[j] = result3['LEIR_space_charge_limit']
            SC_LEIR4_array[j] = result4['LEIR_space_charge_limit']
            
            SC_SPS1_array[j] = result1['SPS_spaceChargeLimit']
            SC_SPS2_array[j] = result2['SPS_spaceChargeLimit']
            SC_SPS3_array[j] = result3['SPS_spaceChargeLimit']
            SC_SPS4_array[j] = result4['SPS_spaceChargeLimit']
            
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
            df.to_csv('../output/csv_tables/charge_state_scan/{}_leir_charge_state_scan{}.csv'.format(ion, output_name))
        ion_dataframes.append(df)
        
        #### PLOTTING - Make figure for all the charge states ####
        fig, ax = plt.subplots(1, 1, figsize = (6,5))
        fig.suptitle(ion, fontsize=20)
        ax.plot(Q_default, Nb0, 'ro', markersize=10.5, alpha=0.8, label='Baseline with default charge state')
        ax.plot(Q_states, Nb1_array, color='blue', linewidth=3, linestyle='-', label='Baseline')
        ax.plot(Q_states, Nb2_array, linestyle='--', color='gold', linewidth=3, label='No PS splitting') #
        ax.plot(Q_states, Nb3_array, linestyle='-.', color='limegreen', linewidth=3, label='LEIR-PS stripping') #
        ax.plot(Q_states, Nb4_array, linestyle='--', color='gray', linewidth=3, label='LEIR-PS stripping, \nno PS splitting') #
        if WG5_intensity[ion] > 0.0:
            ax.axhline(y = WG5_intensity[ion], color='red', label='WG5')
        ax.set_ylabel('LHC bunch intensity')
        ax.set_xlabel('LEIR charge state')
        ax.legend()
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        if save_fig:
            fig.savefig('../output/figures/charge_state_scan/{}_{}_leir_charge_state_scan{}.png'.format(count, ion, output_name), dpi=250)
        plt.close()
        
        #### PLOTTING - Make figure for the LEIR and SPS space charge limits ####
        fig2, ax2 = plt.subplots(1, 1, figsize = (6,5))
        fig2.suptitle(ion, fontsize=20)
        ax2.plot(Q_states, SC_LEIR1_array, color='blue', linewidth=3, linestyle='--', label='LEIR SC limit: Baseline')
        ax2.plot(Q_states, SC_LEIR2_array, linestyle='--', color='gold', linewidth=3, label='LEIR SC limit: No PS splitting') #
        ax2.plot(Q_states, SC_LEIR3_array, linestyle='--', color='limegreen', linewidth=3, label='LEIR SC limit: LEIR-PS stripping') #
        ax2.plot(Q_states, SC_LEIR4_array, linestyle='--', color='gray', linewidth=3, label='LEIR SC limit: LEIR-PS stripping, \nno PS splitting') #
        ax2.plot(Q_states, SC_SPS1_array, color='blue', linewidth=3, linestyle=':', label='SPS SC limit: Baseline')
        ax2.plot(Q_states, SC_SPS2_array, linestyle=':', color='gold', linewidth=3, label='SPS SC limit: No PS splitting') #
        ax2.plot(Q_states, SC_SPS3_array, linestyle=':', color='limegreen', linewidth=3, label='SPS SC limit: LEIR-PS stripping') #
        ax2.plot(Q_states, SC_SPS4_array, linestyle=':', color='gray', linewidth=3, label='SPS SC limit: LEIR-PS stripping, \nno PS splitting') #
        #if WG5_intensity[ion] > 0.0:
        #    ax2.axhline(y = WG5_intensity[ion], color='red', label='WG5')
        ax2.set_ylabel('Space charge limit')
        ax2.set_xlabel('LEIR charge state')
        ax2.set_yscale('log')
        ax2.legend()
        fig2.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        if save_fig:
            fig2.savefig('../output/figures/charge_state_scan/LEIR_SPS_SC_limit_{}_leir_charge_state_scan{}.png'.format(ion, output_name), dpi=250)
        plt.close()
        
        count += 1
        
    return ion_dataframes

if __name__ == '__main__': 
    
    # First check Roderik's case with no reference energy and PS limit  
    dfs_0 = vary_charge_state_and_plot(
                                    output_name='',
                                    consider_PS_space_charge_limit=False, 
                                    use_gammas_ref=False,
                                    save_fig=True
                                    )

    # Then check case with reference gammas - USE WITH CAUTION, HAS IN REALITY TO BE RECALCULATED FOR EVERY NEW CHARGE STATE 
    dfs_1 = vary_charge_state_and_plot(
                                    output_name='_with_reference_gammas_fixed_at_PS_extraction',
                                    consider_PS_space_charge_limit=False, 
                                    use_gammas_ref=True,
                                    save_fig=True 
                                    )
    
    # Compare ration between these two
    #print("\n--------- Comparing with and without reference energies ---------:\n")
    #for i, df in enumerate(dfs_0):
    #    print('\nRatio for {}'.format(ion_data.T.index[i]))
    #    print(dfs_0[i].div(dfs_1[i]))

    # Then check case with reference gammas and PS space charge limit 
    dfs_2 = vary_charge_state_and_plot(
                                    output_name='_with_PS_SC_limit',
                                    consider_PS_space_charge_limit=True, 
                                    use_gammas_ref=False,
                                    save_fig=True 
                                    )