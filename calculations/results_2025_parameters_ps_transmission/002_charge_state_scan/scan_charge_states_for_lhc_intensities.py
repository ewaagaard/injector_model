"""
Main script to calculate final nucleon intensity into the LHC scanning over charge states

This scenario considers only specifically tested charge states for the unstripped species: He, Mg, Ar, O and Kr
Kr is also tested for the stripped scenario: assume peak at Q = 29+ with 25 uA

For Kr, In, Ca, Xe and Pb, use results from Baron's formula 
"""
import matplotlib.pyplot as plt
import pandas as pd 
from injector_model import InjectorChain
import numpy as np
from pathlib import Path
import os
from collections import defaultdict

# Load ion data and initialize for test for bunch intensities 
data_folder = Path(__file__).resolve().parent.joinpath('../../../data').absolute()
os.makedirs('output/figures', exist_ok=True)
os.makedirs('output/output_for_paper', exist_ok=True)
os.makedirs('output/charge_scan_results', exist_ok=True)

# Load possible charge states and currents
full_ion_data = pd.read_csv("{}/Ion_species.csv".format(data_folder), header=0, index_col=0).T
ions_not_stripped = ['He', 'O', 'Mg', 'Ar', 'Kr']
account_for_LEIR_ecooling = True


# Define all relevant scenarios (baseline, stripping, PS splitting, etc) in a function
def calculate_LHC_intensities_all_scenarios_vary_charge_state(
                                            ion_type : str,
                                            Q_dist : pd.DataFrame,
                                            Linac3_current : pd.DataFrame,
                                            account_for_LEIR_ecooling=True,
                                            output_extra_str=''
                                            ):
    
    print('\nIon: {}'.format(ion_type))

    ecool_str = '' if account_for_LEIR_ecooling else '_without_ecooling_limits'

    ## CASE 1: BASELINE (default Pb production)
    output_1 = '1_baseline_{}'.format(output_extra_str)
    injector_chain1 = InjectorChain(PS_splitting = 2,
                                    account_for_LEIR_ecooling=account_for_LEIR_ecooling,
                                    )
                                    
    ## 2: WITHOUT PS SPLITTING
    output_2 ='2_no_PS_splitting_{}'.format(output_extra_str)
    injector_chain2 = InjectorChain(PS_splitting = 1,
                                    account_for_LEIR_ecooling=account_for_LEIR_ecooling,
                                    )
    
    ## 3: WITH PS SPLITTING AND LEIR-PS STRIPPING
    output_3 = '3_LEIR_PS_stripping_{}'.format(output_extra_str)
    injector_chain3 = InjectorChain(PS_splitting = 2,
                                    account_for_LEIR_ecooling=account_for_LEIR_ecooling,
                                    LEIR_PS_strip=True)
    
    ## 4: WITH NO SPLITTING AND LEIR-PS STRIPPING
    output_4 = '4_no_PS_splitting_and_LEIR_PS_stripping_{}'.format(output_extra_str)
    injector_chain4 = InjectorChain(PS_splitting = 1,
                                    account_for_LEIR_ecooling=account_for_LEIR_ecooling,
                                    LEIR_PS_strip=True)

    # Initialize full dicionary for all scenarios
    full_result_df1 = defaultdict(list)
    full_result_df2 = defaultdict(list)
    full_result_df3 = defaultdict(list)
    full_result_df4 = defaultdict(list)

    # Iterate over charge state and Linac3 current
    for i, current in enumerate(Linac3_current):

        # Define new charge state Q before stripping
        Q = Q_dist.iloc[i]

        # Update Linac3/LEIR charge state and Linac3 current in custom ion data
        custom_ion_data = injector_chain1.full_ion_data[ion_type].copy() # copy entry
        custom_ion_data['Q before stripping'] = Q
        custom_ion_data['Linac3 current [uA]'] = current
        print('Q strip = {} has Linac3 current: {} uA'.format(Q, current))

        # Initialize ions in each injector chain scenario
        injector_chain1.init_ion(ion_type=ion_type, ion_data_custom=custom_ion_data)
        df1 = injector_chain1.calculate_LHC_bunch_intensity()

        injector_chain2.init_ion(ion_type=ion_type, ion_data_custom=custom_ion_data)
        df2 = injector_chain2.calculate_LHC_bunch_intensity()

        injector_chain3.init_ion(ion_type=ion_type, ion_data_custom=custom_ion_data)
        df3 = injector_chain3.calculate_LHC_bunch_intensity()

        injector_chain4.init_ion(ion_type=ion_type, ion_data_custom=custom_ion_data)
        df4 = injector_chain4.calculate_LHC_bunch_intensity()

        # Append the values to the corresponding key 
        for key, value in df1.items():
            full_result_df1[key].append(value)
        for key, value in df2.items():
            full_result_df2[key].append(value)
        for key, value in df3.items():
            full_result_df3[key].append(value)
        for key, value in df4.items():
            full_result_df4[key].append(value)       
        
        del df1, df2, df3, df4

    # Combine into big dataframe, save to csv
    df1_all_LEIR_charge_states = pd.DataFrame(full_result_df1).set_index("Q_LEIR")
    df2_all_LEIR_charge_states = pd.DataFrame(full_result_df2).set_index("Q_LEIR")
    df3_all_LEIR_charge_states = pd.DataFrame(full_result_df3).set_index("Q_LEIR")
    df4_all_LEIR_charge_states = pd.DataFrame(full_result_df4).set_index("Q_LEIR")

    df1_all_LEIR_charge_states.T.to_csv("output/charge_scan_results/{}_{}{}_ps_rest_gas.csv".format(ion_type, output_1, ecool_str))
    df2_all_LEIR_charge_states.T.to_csv("output/charge_scan_results/{}_{}{}_ps_rest_gas.csv".format(ion_type, output_2, ecool_str))
    df3_all_LEIR_charge_states.T.to_csv("output/charge_scan_results/{}_{}{}_ps_rest_gas.csv".format(ion_type, output_3, ecool_str))
    df4_all_LEIR_charge_states.T.to_csv("output/charge_scan_results/{}_{}{}_ps_rest_gas.csv".format(ion_type, output_4, ecool_str))




# Scan over ions - scale relative abundance in charge state with Linac3 current
for ion_type in full_ion_data.columns:

    if ion_type in ions_not_stripped:
        strip_state = 'unstripped_currents'
    
        # Unstripped ions after LINAC3 - define path and load LINAC3 current data
        ion_path = '{}/charge_state_scan_currents/{}_{}.parquet'.format(data_folder, ion_type, strip_state)
        df = pd.read_parquet(ion_path)
        print('\nIon type: {}, UNSTRIPPED'.format(ion_type))

        Q_dist = df['Charge state']
        Linac3_current = df['Linac3 current']

        calculate_LHC_intensities_all_scenarios_vary_charge_state(ion_type=ion_type,
                                                                  Q_dist=Q_dist,
                                                                  Linac3_current=Linac3_current,
                                                                  account_for_LEIR_ecooling=account_for_LEIR_ecooling,
                                                                  output_extra_str='UNSTRIPPED')

    if ion_type not in ions_not_stripped or ion_type == 'Kr':
        strip_state = 'stripped_Barons_formula_currents'
    
        # Same for stripped ions after LINAC3
        ion_path = '{}/charge_state_scan_currents/{}_{}.parquet'.format(data_folder, ion_type, strip_state)
        df = pd.read_parquet(ion_path)
        print('\nIon type: {}, STRIPPED'.format(ion_type))

        # Remove not relevant charge states, cannot go above Z
        df = df.drop(df[df['Charge state'] > full_ion_data[ion_type].Z].index)

        Q_dist = df['Charge state']
        Linac3_current = df['Linac3 current']

        calculate_LHC_intensities_all_scenarios_vary_charge_state(ion_type=ion_type,
                                                            Q_dist=Q_dist,
                                                            Linac3_current=Linac3_current,
                                                            account_for_LEIR_ecooling=account_for_LEIR_ecooling,
                                                            output_extra_str='STRIPPED')
