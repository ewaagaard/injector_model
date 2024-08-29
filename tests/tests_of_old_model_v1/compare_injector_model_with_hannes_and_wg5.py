#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script to calculate final nucleon intensity into the LHC considering linear space charge limit in LEIR and SPS,
but not yet in PS 
"""
import matplotlib.pyplot as plt
import pandas as pd 
from injector_model import InjectorChain
import numpy as np

# Load ion data and initialize for test for bunch intensities 
ion_data = pd.read_csv("../data/Ion_species.csv", sep=';', header=0, index_col=0).T
ion_type = 'Pb'
injector_chain = InjectorChain(ion_type, ion_data, account_for_SPS_transmission=False)
df_Nb = injector_chain.simulate_SpaceCharge_intensity_limit_all_ions()

# Compare to reference intensities
ref_Table_SPS = pd.read_csv('../data/test_and_benchmark_data/SPS_final_intensities_WG5_and_Hannes.csv', delimiter=';', index_col=0)
ref_Table_LEIR = pd.read_csv('../data/test_and_benchmark_data/LEIR_final_intensities_Nicolo.csv', delimiter=';', index_col=0)

print("\nRelevant comparison cases:\n")
print('\nSPS intensity vs WG5 ratrio:\n')
print(df_Nb['Nb_SPS']/ref_Table_SPS['WG5 Intensity'])
print('\nSPS intensity Hannes ratrio:\n')
print(df_Nb['Nb_SPS']/ref_Table_SPS['Hannes Intensity '])
print('\nLEIR intensity Hannes ratrio:\n')
print(df_Nb['Nb_LEIR']/ref_Table_LEIR['Nicolo Intensity'])