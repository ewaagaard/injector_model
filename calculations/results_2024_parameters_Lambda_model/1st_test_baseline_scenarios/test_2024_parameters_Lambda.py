"""
Test script to run Injector Model version 2025 (i.e. with 2024 Pb ion run as input) with updated model parameters
"""
import pandas as pd
import matplotlib.pyplot as plt
from injector_model import InjectorChain
import numpy as np
import os

# First test baseline case with default Pb production - 2024
injector_chain1 = InjectorChain()
result = injector_chain1.calculate_LHC_bunch_intensity()
df = pd.Series(result)

# Print resulting values
print('Results [ions per bunch]:')
for i, item in enumerate(df.items()):
    if type(item[1]) == np.float64:
        ions_to_charge = 54 if i<22 else 82
        print('{:25}: {:.2e}'.format(item[0], item[1]))
    else:
        print('{:25}: {}'.format(item[0], item[1]))

# Scan over all ions 
df1 = injector_chain1.calculate_LHC_bunch_intensity_all_ion_species(save_csv=True, output_name = '1_Baseline_with_2024_Pb_ion_parameters')

# Then run similar tests, but with no electron cooling limits
injector_chain2 = InjectorChain(account_for_LEIR_ecooling=False)
df2 = injector_chain2.calculate_LHC_bunch_intensity_all_ion_species(save_csv=True, output_name = '2_Baseline_with_2024_Pb_ion_parameters_no_ecooling_limits')
