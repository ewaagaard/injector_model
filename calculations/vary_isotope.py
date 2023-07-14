#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to vary isotope to find optimum state for maximum injected LHC intensity 
- for unnatural isotopes, scan with A of the initial mass to understand mass dependence 
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
full_isotope_data = pd.read_csv('../data/Full_isotope_data.csv', index_col=0)

# Stable isotope data
He_isotopes = np.array([3., 4.])
O_isotopes = np.array([16., 17., 18.])
Ar_isotopes = np.array([36., 38., 40.])
Ca_isotopes = np.array([40., 42., 43., 44., 46., 48.])
Kr_isotopes = np.array([78., 80., 82., 83., 84., 86.])
In_isotopes = np.array([113., 115.])
Xe_isotopes = np.array([124., 126., 128., 129., 130., 131., 132, 134., 136])
Pb_isotopes = np.array([204., 206., 207., 208.])

all_isotopes = [He_isotopes, O_isotopes, Ar_isotopes, Ca_isotopes, Kr_isotopes, In_isotopes, Xe_isotopes, Pb_isotopes]

# Compare to reference intensities - WG5 and Roderik
ref_Table_SPS = pd.read_csv('../data/test_and_benchmark_data/SPS_final_intensities_WG5_and_Hannes.csv', delimiter=';', index_col=0)
WG5_intensity = ref_Table_SPS['WG5 Intensity']

# Define all relevant scenarios (baseline, stripping, PS splitting, etc) in a function
def calculate_LHC_intensities_all_scenarios_vary_isotope(
                                            A,
                                            ion_type, 
                                            consider_PS_space_charge_limit,
                                            use_gammas_ref
                                            ):
    A_default = ion_data[ion_type]['A']
    mass_default = ion_data[ion_type]['mass [GeV]']
    
    ## CASE 1: BASELINE (default Pb production)
    injector_chain1 = InjectorChain(ion_type, 
                                    ion_data, 
                                    nPulsesLEIR = 0,
                                    LEIR_bunches = 2,
                                    PS_splitting = 2,
                                    consider_PS_space_charge_limit = consider_PS_space_charge_limit,
                                    use_gammas_ref = use_gammas_ref
                                    )
    injector_chain1.A = A  # update mass number
    injector_chain1.mass_GeV = mass_default * (A / A_default)  # update mass according to simple scaling 
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
    injector_chain2.A = A # update mass number
    injector_chain2.mass_GeV = mass_default * (A / A_default)  # update mass according to simple scaling 
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
    
    injector_chain3.A = A  # update mass number
    injector_chain3.mass_GeV = mass_default * (A / A_default)  # update mass according to simple scaling 
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
    injector_chain4.A = A  # update mass number
    injector_chain4.mass_GeV = mass_default * (A / A_default)  # update mass according to simple scaling 
    result4 = injector_chain4.calculate_LHC_bunch_intensity()
    
    return result1, result2, result3, result4

# Function to vary isotope and return a dictionary 
def vary_isotope_and_plot(
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
    #fig0.suptitle("Isotope Scan", fontsize=20)
    
    # Iterate over all ion species 
    count = 1
    for ion, row in ion_data.T.iterrows():
        
        print('\nVarying mass number for {}'.format(ion))
        
        A_default = row['A']
        A_states = all_isotopes[count-1] # array of stable ions
        
        # Create empty array for all the LHC bunch intensities, LEIR SC limit and SPS SC limit  
        Nb1_array, Nb2_array, Nb3_array, Nb4_array = np.zeros(len(A_states)), np.zeros(len(A_states)), np.zeros(len(A_states)), np.zeros(len(A_states)) 
        SC_SPS1_array, SC_SPS2_array, SC_SPS3_array, SC_SPS4_array = np.zeros(len(A_states)), np.zeros(len(A_states)), np.zeros(len(A_states)), np.zeros(len(A_states)) 
        SC_LEIR1_array, SC_LEIR2_array, SC_LEIR3_array, SC_LEIR4_array = np.zeros(len(A_states)), np.zeros(len(A_states)), np.zeros(len(A_states)), np.zeros(len(A_states))
        gammas_SPS1_array, gammas_SPS2_array, gammas_SPS3_array, gammas_SPS4_array = np.zeros(len(A_states)), np.zeros(len(A_states)), np.zeros(len(A_states)), np.zeros(len(A_states)) 
        
        # First check default intensity from standard isotope
        result01, result02, result03, result04 = calculate_LHC_intensities_all_scenarios_vary_isotope(
                                                                                                        A_default,
                                                                                                        ion, 
                                                                                                        consider_PS_space_charge_limit,
                                                                                                        use_gammas_ref,
                                                                                                        )
        Nb0 = result01['LHC_ionsPerBunch'] # LHC bunch intensity for default baseline scenario and isotope
        
        # Iterate over all the A_states 
        for j, A in enumerate(A_states):
            result1, result2, result3, result4 = calculate_LHC_intensities_all_scenarios_vary_isotope(
                                                                                                        A,
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
                'A_state': A_states,
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
        df = df.set_index(['A_state'])
        if save_fig:
            df.to_csv('../output/csv_tables/isotope_scan/isotope_scan_{}_{}.csv'.format(ion, output_name))
        ion_dataframes.append(df)
        
        #### PLOTTING - Make figure for all the isotopes ####
        fig, ax = plt.subplots(1, 1, figsize = (6,5))
        fig.suptitle(ion, fontsize=20)
        if WG5_intensity[ion] > 0.0:
            ax.axhline(y = WG5_intensity[ion], color='red', label='WG5')
        ax.plot(A_default, Nb0, 'ro', markersize=14.5, alpha=0.8, label='Baseline with default isotope')
        ax.plot(A_states, Nb1_array, color='blue', marker='o', linewidth=3, linestyle='-', label='Baseline')
        ax.plot(A_states, Nb2_array, linestyle='--', marker='o', color='gold', linewidth=3, label='No PS splitting') #
        ax.plot(A_states, Nb3_array, linestyle='-.', marker='o', color='limegreen', linewidth=3, label='LEIR-PS stripping') #
        ax.plot(A_states, Nb4_array, linestyle='--', marker='o', color='gray', linewidth=3, label='LEIR-PS stripping, \nno PS splitting') #
        ax.set_ylabel('LHC bunch intensity')
        ax.set_xlabel('Mass number A')
        ax.legend()
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        if save_fig:
            fig.savefig('../output/figures/isotope_scan/isotope_scan_{}_{}_{}.png'.format(count, ion, output_name), dpi=250)
        plt.close()
        
        #### PLOTTING - Make figure for the LEIR and SPS space charge limits ####
        fig2, ax2 = plt.subplots(1, 1, figsize = (6,5))
        fig2.suptitle(ion, fontsize=20)
        ax2.plot(A_states, SC_LEIR1_array, color='blue', marker='o', linewidth=3, linestyle='-', label='LEIR SC limit: Baseline')
        #ax2.plot(A_states, SC_LEIR2_array, linestyle='--', color='gold', linewidth=3, label='LEIR SC limit: No PS splitting') #
        #ax2.plot(A_states, SC_LEIR3_array, linestyle='--', color='limegreen', linewidth=3, label='LEIR SC limit: LEIR-PS stripping') #
        #ax2.plot(A_states, SC_LEIR4_array, linestyle='--', color='gray', linewidth=3, label='LEIR SC limit: LEIR-PS stripping, \nno PS splitting') #
        ax2.plot(A_states, SC_SPS1_array, color='blue', marker='o', linewidth=3, linestyle=':', label='SPS SC limit: Baseline')
        #ax2.plot(A_states, SC_SPS2_array, linestyle=':', color='gold', linewidth=3, label='SPS SC limit: No PS splitting') #
        #ax2.plot(A_states, SC_SPS3_array, linestyle=':', color='limegreen', linewidth=3, label='SPS SC limit: LEIR-PS stripping') #
        #ax2.plot(A_states, SC_SPS4_array, linestyle=':', color='gray', linewidth=3, label='SPS SC limit: LEIR-PS stripping, \nno PS splitting') #
        #if WG5_intensity[ion] > 0.0:
        #    ax2.axhline(y = WG5_intensity[ion], color='red', label='WG5')
        ax2.set_ylabel('Space charge limit')
        ax2.set_xlabel('Mass number A')
        #ax2.set_yscale('log')
        ax2.legend()
        fig2.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        if save_fig:
            fig2.savefig('../output/figures/isotope_scan/isotope_scan_LEIR_SPS_SC_limit_{}_{}.png'.format(ion, output_name), dpi=250)
        plt.close()
        
        # Also fill in the big combined subplot
        row3 = (count-1) // num_cols  # Row index
        col3 = (count-1) % num_cols  # Column index
        ax3 = axs[row3, col3]  # Select the current subplot

        # Plot the data for the current ion
        ax3.plot(A_default, Nb0, 'ro', markersize=11, alpha=0.8, label='Baseline with default isotope state')
        if WG5_intensity[ion] > 0.0:
            ax3.axhline(y=WG5_intensity[ion], color='red', label='WG5')
        ax3.plot(A_states, Nb1_array,  marker='o', color='blue', linewidth=3, linestyle='-', label='Baseline')
        ax3.plot(A_states, Nb2_array,  marker='o', linestyle='--', color='gold', linewidth=3, label='No PS splitting')
        ax3.plot(A_states, Nb3_array,  marker='o', linestyle='-.', color='limegreen', linewidth=3, label='LEIR-PS stripping')
        ax3.plot(A_states, Nb4_array,  marker='o', linestyle='--', color='gray', linewidth=3, label='LEIR-PS stripping, \nno PS splitting')
        ax3.set_title(ion)  # Set the ion name as the title for the current subplot
        
        # Add legend in oxygen plot
        if count == 2:
            ax3.legend(fontsize=6)
        
        count += 1

    # Combined figure - Share y-axes for the same row
    # Share x-label for the same column
    for col in range(num_cols):
        axs[-1, col].set_xlabel('Mass number A', fontsize=13)
    
    # Share y-label for the same row
    for row in axs:
        row[0].set_ylabel('LHC bunch intensity', fontsize=13)
    
    fig0.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)    
    # Save the combined figure
    if save_fig:
        fig0.savefig('../output/figures/isotope_scan/combined_isotope_scan{}.png'.format(output_name), dpi=250)
    plt.close()    
    
    return ion_dataframes

if __name__ == '__main__': 
    
    # First check Roderik's case with no reference energy and PS limit  
    print('\nTesting without PS space charge limit... \n')
    dfs_0 = vary_isotope_and_plot(
                                    output_name='',
                                    consider_PS_space_charge_limit=False, 
                                    use_gammas_ref=False,
                                    save_fig=True
                                    )


    # Then check case with reference gammas and PS space charge limit
    print('\nTesting with PS space charge limit... \n')
    dfs_1 = vary_isotope_and_plot(
                                    output_name='_with_PS_SC_limit',
                                    consider_PS_space_charge_limit=True, 
                                    use_gammas_ref=False,
                                    save_fig=True 
                                    )
    
    # Store best isotope and injected LHC intensity 
    A_best_1_array = np.zeros(len(ion_data.T.index))
    A_best_2_array = np.zeros(len(ion_data.T.index))
    A_best_3_array = np.zeros(len(ion_data.T.index))
    A_best_4_array = np.zeros(len(ion_data.T.index))
    
    Nb_best_1_array = np.zeros(len(ion_data.T.index))
    Nb_best_2_array = np.zeros(len(ion_data.T.index))
    Nb_best_3_array = np.zeros(len(ion_data.T.index))
    Nb_best_4_array = np.zeros(len(ion_data.T.index))
    
    # Also considering PS space charge
    A_best_1_array_ps_sc = np.zeros(len(ion_data.T.index))
    A_best_2_array_ps_sc = np.zeros(len(ion_data.T.index))
    A_best_3_array_ps_sc = np.zeros(len(ion_data.T.index))
    A_best_4_array_ps_sc = np.zeros(len(ion_data.T.index))
    
    Nb_best_1_array_ps_sc = np.zeros(len(ion_data.T.index))
    Nb_best_2_array_ps_sc = np.zeros(len(ion_data.T.index))
    Nb_best_3_array_ps_sc = np.zeros(len(ion_data.T.index))
    Nb_best_4_array_ps_sc = np.zeros(len(ion_data.T.index))
    
    i = 0
    for ion, row in ion_data.T.iterrows():
        
        #### First check without PS space charge ####
        # Find the best ion species for LHC
        
        A_best_1_array[i] = dfs_0[i]['Nb0_1_Baseline'].idxmax()
        A_best_2_array[i] = dfs_0[i]['Nb0_2_No_PS_split'].idxmax()
        A_best_3_array[i] = dfs_0[i]['Nb0_3_LEIR_PS_strip'].idxmax()
        A_best_4_array[i] = dfs_0[i]['Nb0_4_LEIR_PS_strip_and_no_PS_split'].idxmax()
        
        Nb_best_1_array[i] = dfs_0[i]['Nb0_1_Baseline'].max()
        Nb_best_2_array[i] = dfs_0[i]['Nb0_2_No_PS_split'].max()
        Nb_best_3_array[i] = dfs_0[i]['Nb0_3_LEIR_PS_strip'].max()
        Nb_best_4_array[i] = dfs_0[i]['Nb0_4_LEIR_PS_strip_and_no_PS_split'].max()
    
        #### Then check PS space charge ####
        
        A_best_1_array_ps_sc[i] = dfs_1[i]['Nb0_1_Baseline'].idxmax()
        A_best_2_array_ps_sc[i] = dfs_1[i]['Nb0_2_No_PS_split'].idxmax()
        A_best_3_array_ps_sc[i] = dfs_1[i]['Nb0_3_LEIR_PS_strip'].idxmax()
        A_best_4_array_ps_sc[i] = dfs_1[i]['Nb0_4_LEIR_PS_strip_and_no_PS_split'].idxmax()
        
        Nb_best_1_array_ps_sc[i] = dfs_1[i]['Nb0_1_Baseline'].max()
        Nb_best_2_array_ps_sc[i] = dfs_1[i]['Nb0_2_No_PS_split'].max()
        Nb_best_3_array_ps_sc[i] = dfs_1[i]['Nb0_3_LEIR_PS_strip'].max()
        Nb_best_4_array_ps_sc[i] = dfs_1[i]['Nb0_4_LEIR_PS_strip_and_no_PS_split'].max()  
    
        i += 1
        
    # Create dictionaries with results 
    dict_best_ions = {'1_A_best': A_best_1_array,
                      '1_Nb_best': Nb_best_1_array,
                      '2_A_best': A_best_2_array,
                      '2_Nb_best': Nb_best_2_array,
                      '3_A_best': A_best_3_array,
                      '3_Nb_best': Nb_best_3_array,
                      '4_A_best': A_best_4_array,
                      '4_Nb_best': Nb_best_4_array,
                      }
    df_best_ions = pd.DataFrame(dict_best_ions)
    df_best_ions = df_best_ions.set_index(ion_data.T.index)
    df_best_ions.to_csv('../output/csv_tables/isotope_scan/best_isotope/best_Nb_isotope_scan.csv', float_format='%e')
    
    dict_best_ions_ps_sc = {'1_A_best': A_best_1_array_ps_sc,
                      '1_Nb_best': Nb_best_1_array_ps_sc,
                      '2_A_best': A_best_2_array_ps_sc,
                      '2_Nb_best': Nb_best_2_array_ps_sc,
                      '3_A_best': A_best_3_array_ps_sc,
                      '3_Nb_best': Nb_best_3_array_ps_sc,
                      '4_A_best': A_best_4_array_ps_sc,
                      '4_Nb_best': Nb_best_4_array_ps_sc,
                      }
    df_best_ions_ps_sc = pd.DataFrame(dict_best_ions_ps_sc)
    df_best_ions_ps_sc = df_best_ions_ps_sc.set_index(ion_data.T.index)
    df_best_ions_ps_sc.to_csv('../output/csv_tables/isotope_scan/best_isotope/best_Nb_isotope_scan_with_PC_SC_limit.csv', float_format='%e')