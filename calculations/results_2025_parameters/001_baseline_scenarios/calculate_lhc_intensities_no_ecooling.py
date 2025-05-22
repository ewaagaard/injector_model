"""
Main script to calculate final nucleon intensity into the LHC with new injector model version
Includes:
- Full space charge limit
- Electron cooling (set to False)
"""
import matplotlib.pyplot as plt
import pandas as pd 
from injector_model import InjectorChain
import numpy as np
from pathlib import Path
import os
import re

# Load ion data and initialize for test for bunch intensities 
data_folder = Path(__file__).resolve().parent.joinpath('../../../data').absolute()
os.makedirs('output_no_ecooling', exist_ok=True)


                                           
## CASE 1: BASELINE (default Pb production)
output_1 = '1_baseline'
injector_chain1 = InjectorChain(LEIR_bunches = 2,
                            PS_splitting = 2,
                            LEIR_PS_strip=False,
                            account_for_LEIR_ecooling=False)
result = injector_chain1.calculate_LHC_bunch_intensity()

# Calculate LHC bunch intensity for all ions
df = injector_chain1.calculate_LHC_bunch_intensity_all_ion_species(save_csv=True, output_name = output_1)



df.to_csv('output_no_ecooling/{}_for_paper.csv'.format(output_1), index=True)
    
