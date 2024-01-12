#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test baseline scenario for Magnesium (Mg) 
"""
import matplotlib.pyplot as plt
import pandas as pd 
from injector_model import InjectorChain
import numpy as np
import scipy

# Load ion data
ion_data = pd.read_csv("../data/Ion_species.csv", header=0, index_col=0).T

# Add Mg data  - assume 12_24_Mg - stable isotope with 80% prevalance
# --> assume 
# - same LINAC3 pulse length as all other species
# - conservatively same LINAC3 current as 18Ar (60 uA)
# - 90% stripping efficiency (as He, O, Ar and Ca)
ion_data['Mg'] = ion_data['Ar']

# Update other parameters - also find Magnesium mass in atomic units
ion_data['Mg'].Z = 12.0
ion_data['Mg'].A = 24.0
ion_data['Mg']['Q before stripping'] = 6.0 # Reyes had assumed 6+ or 7+ before stripping
ion_data['Mg']['mass [GeV]'] = 23.985041697 * scipy.constants.physical_constants['atomic mass unit-electron volt relationship'][0] * 1e-9
ion_type = 'Mg'

# Also Mg7+
ion_data['Mg_7'] = ion_data['Mg']
ion_data['Mg_7']['Q before stripping'] = 7.0

## CASE 1: BASELINE (default Pb production)
injector_chain1 = InjectorChain(ion_type, 
                                ion_data, 
                                nPulsesLEIR = 0,
                                LEIR_bunches = 2,
                                PS_splitting = 2,
                                consider_PS_space_charge_limit = True
                                )
result = injector_chain1.calculate_LHC_bunch_intensity()
    
# Calculate LHC bunch intensity for all ions
df = injector_chain1.calculate_LHC_bunch_intensity_all_ion_species(save_csv=True, output_name = '1_baseline_with_Mg')
