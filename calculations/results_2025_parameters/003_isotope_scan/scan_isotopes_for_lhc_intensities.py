"""
Main script to calculate final nucleon intensity into the LHC scanning over stable isotopes
- simple scale up mass acording to number of nucleons, mostly true
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
os.makedirs('output/isotope_scan_results', exist_ok=True)

# Load default ion data
full_ion_data = pd.read_csv("{}/Ion_species.csv".format(data_folder), header=0, index_col=0).T
account_for_LEIR_ecooling = True

# Stable isotope data
He_isotopes = np.array([3., 4.])
O_isotopes = np.array([16., 17., 18.])
Mg_isotopes = np.array([24., 25., 26.])
Ar_isotopes = np.array([36., 38., 40.])
Ca_isotopes = np.array([40., 42., 43., 44., 46., 48.])
Kr_isotopes = np.array([78., 80., 82., 83., 84., 86.])
In_isotopes = np.array([113., 115.])
Xe_isotopes = np.array([124., 126., 128., 129., 130., 131., 132, 134., 136])
Pb_isotopes = np.array([204., 206., 207., 208.])

all_isotopes = [He_isotopes, O_isotopes, Mg_isotopes, Ar_isotopes, Ca_isotopes, Kr_isotopes, In_isotopes, Xe_isotopes, Pb_isotopes]


# Define all relevant scenarios (baseline, stripping, PS splitting, etc) in a function
def calculate_LHC_intensities_all_scenarios_vary_isotope(
                                            ion_type : str,
                                            A_array : np.ndarray,
                                            account_for_LEIR_ecooling=account_for_LEIR_ecooling,
                                            output_extra_str=''
                                            ):
    
    print('\nIon: {}'.format(ion_type))

    ecool_str = '' if account_for_LEIR_ecooling else '_without_ecooling_limits'

    ## CASE 1: BASELINE (default Pb production)
    output_1 = '1_baseline{}'.format(output_extra_str)
    injector_chain1 = InjectorChain(PS_splitting = 2, account_for_LEIR_ecooling=account_for_LEIR_ecooling)
                                    
    ## 2: WITHOUT PS SPLITTING
    output_2 ='2_no_PS_splitting{}'.format(output_extra_str)
    injector_chain2 = InjectorChain(PS_splitting = 1, account_for_LEIR_ecooling=account_for_LEIR_ecooling)
    
    ## 3: WITH PS SPLITTING AND LEIR-PS STRIPPING
    output_3 = '3_LEIR_PS_stripping{}'.format(output_extra_str)
    injector_chain3 = InjectorChain(PS_splitting = 2,
                                    account_for_LEIR_ecooling=account_for_LEIR_ecooling,
                                    LEIR_PS_strip=True)
    
    ## 4: WITH NO SPLITTING AND LEIR-PS STRIPPING
    output_4 = '4_no_PS_splitting_and_LEIR_PS_stripping{}'.format(output_extra_str)
    injector_chain4 = InjectorChain(PS_splitting = 1,
                                    account_for_LEIR_ecooling=account_for_LEIR_ecooling,
                                    LEIR_PS_strip=True)

    # Initialize full dicionary for all scenarios
    full_result_df1 = defaultdict(list)
    full_result_df2 = defaultdict(list)
    full_result_df3 = defaultdict(list)
    full_result_df4 = defaultdict(list)

    # Iterate over mass number A for each stable isotope
    for i, A_new in enumerate(A_array):

        # Update mass number and mass custom ion data
        custom_ion_data = injector_chain1.full_ion_data[ion_type].copy() # copy entry
        custom_ion_data['A'] = A_new
        custom_ion_data['mass [GeV]'] = injector_chain1.full_ion_data[ion_type]['mass [GeV]'] * (A_new / injector_chain1.full_ion_data[ion_type]['A'])
        print('{}\n'.format(custom_ion_data))
        
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
    df1_all_isotopes = pd.DataFrame(full_result_df1).set_index("Q_LEIR")
    df2_all_isotopes = pd.DataFrame(full_result_df2).set_index("Q_LEIR")
    df3_all_isotopes = pd.DataFrame(full_result_df3).set_index("Q_LEIR")
    df4_all_isotopes = pd.DataFrame(full_result_df4).set_index("Q_LEIR")

    df1_all_isotopes.T.to_csv("output/isotope_scan_results/{}_{}{}.csv".format(ion_type, output_1, ecool_str))
    df2_all_isotopes.T.to_csv("output/isotope_scan_results/{}_{}{}.csv".format(ion_type, output_2, ecool_str))
    df3_all_isotopes.T.to_csv("output/isotope_scan_results/{}_{}{}.csv".format(ion_type, output_3, ecool_str))
    df4_all_isotopes.T.to_csv("output/isotope_scan_results/{}_{}{}.csv".format(ion_type, output_4, ecool_str))




# Scan over isotopes for all ion
for i, ion_type in enumerate(full_ion_data.columns):


    calculate_LHC_intensities_all_scenarios_vary_isotope(ion_type=ion_type,
                                                         A_array=all_isotopes[i],
                                                         account_for_LEIR_ecooling=account_for_LEIR_ecooling)

