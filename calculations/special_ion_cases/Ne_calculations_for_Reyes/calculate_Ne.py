#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test baseline scenario for Neon (20Ne10)
"""
import matplotlib.pyplot as plt
import pandas as pd 
from injector_model import InjectorChain
import numpy as np
import scipy

# Load ion data
ion_data = pd.read_csv("../../data/Ion_species.csv", header=0, index_col=0).T

# Assumptions
# - we use 20Ne10 as isotope
# - same LINAC3 pulse length as all other species
# - 90% stripping efficiency (as He, O, Ar and Ca)
# - Q_LEIR and Q_PS = 5+, just like Reyes had assumed
# - single injection into LEIR, i.e. the EARLY beam, since the Neon run will be a pilot run too

# Test 4 cases of different Linac3 current 
# 1.     70 microAmps (uA) out of Linac3, like for Oxygen
# 2.     60 uA, like for Argon in 2015
# 3.     50 uA
# 4.     40 uA as it was estimated from the Krypton test last year

# Make new Ne entry copying values for Mg
ion_data['Ne'] = ion_data['O']

# Remove all rows we are not interested in - only keep O, Pb and all cases of Ne
del ion_data['He'], ion_data['Ar'], ion_data['Ca'], ion_data['Kr'], ion_data['In'], ion_data['Xe'] 

# Update other parameters - also find Magnesium mass in atomic units
ion_data['Ne'].Z = 10.0
ion_data['Ne'].A = 20.0
ion_data['Ne']['str'] = 'Ne'
ion_data['Ne']['Q before stripping'] = 5.0 # 
ion_data['Ne']['mass [GeV]'] = 19.992440 * scipy.constants.physical_constants['atomic mass unit-electron volt relationship'][0] * 1e-9
ion_type = 'Ne'

# Make new entries with different LINAC currents
Linac_uA = [60.0, 50.0, 40.0]
strs = ['Ne_2', 'Ne_3', 'Ne_4']
for i, u in enumerate(Linac_uA):
    ion_data[strs[i]] = ion_data['Ne']
    ion_data[strs[i]]['Linac3 current [uA]'] = u

print(ion_data)

## CASE 1: BASELINE (like default Pb production)
injector_chain1 = InjectorChain(ion_type, 
                                ion_data, 
                                nPulsesLEIR = 1,
                                LEIR_bunches = 1,
                                PS_splitting = 2,
                                consider_PS_space_charge_limit = True
                                )
result = injector_chain1.calculate_LHC_bunch_intensity()
    
# Calculate LHC bunch intensity for all ions
df = injector_chain1.calculate_LHC_bunch_intensity_all_ion_species(save_csv=True, output_name = '1_baseline_with_Ne')

## CASE 2: NO PS BUNCH SPLITTING
injector_chain2 = InjectorChain(ion_type, 
                                ion_data, 
                                nPulsesLEIR = 1,
                                LEIR_bunches = 1,
                                PS_splitting = 1,
                                consider_PS_space_charge_limit = True
                                )
result2 = injector_chain2.calculate_LHC_bunch_intensity()
    
# Calculate LHC bunch intensity for all ions
df2 = injector_chain2.calculate_LHC_bunch_intensity_all_ion_species(save_csv=True, output_name = '2_no_PS_splitting_with_Ne')


######### USE REFERENCE ENERGIES without simplified gamma formula #########

## CASE 3: BASELINE (like default Pb production) with reference energy
injector_chain3 = InjectorChain(ion_type, 
                                ion_data, 
                                nPulsesLEIR = 1,
                                LEIR_bunches = 1,
                                PS_splitting = 2,
                                consider_PS_space_charge_limit = True,
                                use_gammas_ref=True
                                )
result3 = injector_chain3.calculate_LHC_bunch_intensity()
    
# Calculate LHC bunch intensity for all ions
df3 = injector_chain3.calculate_LHC_bunch_intensity_all_ion_species(save_csv=True, output_name = '3_baseline_with_Ne_gamma_ref')

## CASE 4: NO PS BUNCH SPLITTING with reference energy
injector_chain4 = InjectorChain(ion_type, 
                                ion_data, 
                                nPulsesLEIR = 1,
                                LEIR_bunches = 1,
                                PS_splitting = 1,
                                consider_PS_space_charge_limit = True,
                                use_gammas_ref=True
                                )
result4 = injector_chain4.calculate_LHC_bunch_intensity()
    
# Calculate LHC bunch intensity for all ions
df4 = injector_chain4.calculate_LHC_bunch_intensity_all_ion_species(save_csv=True, output_name = '4_no_PS_splitting_with_Ne_gamma_ref')