"""
Test 4 scenarios with 100% vs 98% slip-stacking - observe the difference
"""
import pandas as pd 
from injector_model import InjectorChain
import numpy as np

    
####### CASE 1: BASELINE (default Pb production) #######
injector_chain1 = InjectorChain(LEIR_bunches = 2,
                                PS_splitting = 2,
                                account_for_LEIR_ecooling=True
                                )
result = injector_chain1.calculate_LHC_bunch_intensity()
df = injector_chain1.calculate_LHC_bunch_intensity_all_ion_species(save_csv=False)

# Update SPS slip-stacking transmission
injector_chain1.SPS_slipstacking_transmission = 0.98
df_new = injector_chain1.calculate_LHC_bunch_intensity_all_ion_species(save_csv=False)



####### 2: NO PS SPLITTING #######
injector_chain2 = InjectorChain(LEIR_bunches = 2,
                                PS_splitting = 1,
                                account_for_LEIR_ecooling=True
                                )
df2 = injector_chain2.calculate_LHC_bunch_intensity_all_ion_species(save_csv=False)

# Update SPS slip-stacking transmission
injector_chain2.SPS_slipstacking_transmission = 0.98
df2_new = injector_chain2.calculate_LHC_bunch_intensity_all_ion_species(save_csv=False)

####### 3: WITH PS SPLITTING AND LEIR-PS STRIPPING #######
injector_chain3 = InjectorChain(LEIR_bunches = 2,
                                PS_splitting = 2,
                                account_for_LEIR_ecooling=True,
                                LEIR_PS_strip=True)

df3 = injector_chain3.calculate_LHC_bunch_intensity_all_ion_species(save_csv=False)

# Update SPS slip-stacking transmission
injector_chain3.SPS_slipstacking_transmission = 0.98
df3_new = injector_chain3.calculate_LHC_bunch_intensity_all_ion_species(save_csv=False)


####### 4: WITH NO SPLITTING AND LEIR-PS STRIPPING #######
injector_chain4 = InjectorChain(LEIR_bunches = 2,
                                PS_splitting = 1,
                                account_for_LEIR_ecooling=True,
                                LEIR_PS_strip=True
                                )
df4 = injector_chain4.calculate_LHC_bunch_intensity_all_ion_species(save_csv=False)
    
# Update SPS slip-stacking transmission
injector_chain4.SPS_slipstacking_transmission = 0.98
df4_new = injector_chain4.calculate_LHC_bunch_intensity_all_ion_species(save_csv=False)

## Check injected LHC bunch intensity in different scenarios:
print('\nRelative decrease injected LHC ions per bunch: 98% slip-stacking transmission vs 100%:')
print('Scenario 1:\n{}'.format(df_new['LHC_ionsPerBunch'] / df['LHC_ionsPerBunch']))
print('Scenario 2:\n{}'.format(df2_new['LHC_ionsPerBunch'] / df2['LHC_ionsPerBunch']))
print('Scenario 3:\n{}'.format(df3_new['LHC_ionsPerBunch'] / df3['LHC_ionsPerBunch']))
print('Scenario 4:\n{}'.format(df4_new['LHC_ionsPerBunch'] / df4['LHC_ionsPerBunch']))