#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to compare IBS growth rates of Pb and O ions from the analytical IBS module with values from 
Bartosik and John 2021 report: (https://cds.cern.ch/record/2749453)
"""
import pandas as pd

# Load dataframes of tune shifts and IBS growth rates - only take values at injection
df_Pb = pd.read_csv('../calculations/full_sc_integral_and_ibs/output/Pb_tune_shifts_over_gammas.csv')
df_O = pd.read_csv('../calculations/full_sc_integral_and_ibs/output/O_tune_shifts_over_gammas.csv')
df_Pb_inj = df_Pb.iloc[0]
df_O_inj = df_O.iloc[0]

# Bartosik and John's IBS growth rates from paper
dict_ref_O = {
            'O_IBS_LEIR_X' : 0.0631,
            'O_IBS_LEIR_Y' : 0.0098,
            'O_IBS_PS_X' : 0.0005,
            'O_IBS_PS_Y' : 0.0073,
            'O_IBS_SPS_X' : 0.0014,
            'O_IBS_SPS_Y' : 0.0042
            }

dict_ref_Pb = {
            'Pb_IBS_LEIR_X' : 1.4712,  # should it be 1 or 2
            'Pb_IBS_LEIR_Y' : 0.1967,
            'Pb_IBS_PS_X' : 0.0150,
            'Pb_IBS_PS_Y' : 0.1080,
            'Pb_IBS_SPS_X' : 0.0044,
            'Pb_IBS_SPS_Y' : 0.0145
            }


df_O_ref = pd.DataFrame.from_dict(dict_ref_O, orient='index', columns=['Reference Value'])
df_Pb_ref = pd.DataFrame.from_dict(dict_ref_Pb, orient='index', columns=['Reference Value'])

# Make dataframe of IBS growth rate fractions fraction
index = ['IBS_LEIR_X', 'IBS_LEIR_Y', 'IBS_PS_X', 'IBS_PS_Y', 'IBS_SPS_X', 'IBS_SPS_Y']
df_ref_frac = df_O_ref.divide(df_Pb_ref.values)
df_ref_frac.index = index
df_ref_frac.index.name = 'Frac O / Pb'

# Extract relevant IBS growth rates for O and Pb
df_O_IBS = pd.DataFrame(df_O_inj[['IBS_LEIR_X', 'IBS_LEIR_Y', 'IBS_PS_X', 'IBS_PS_Y', 'IBS_SPS_X', 'IBS_SPS_Y']], 
                        index = ['IBS_LEIR_X', 'IBS_LEIR_Y', 'IBS_PS_X', 'IBS_PS_Y', 'IBS_SPS_X', 'IBS_SPS_Y']).transpose()
df_Pb_IBS = pd.DataFrame(df_Pb_inj[['IBS_LEIR_X', 'IBS_LEIR_Y', 'IBS_PS_X', 'IBS_PS_Y', 'IBS_SPS_X', 'IBS_SPS_Y']],
                         index = ['IBS_LEIR_X', 'IBS_LEIR_Y', 'IBS_PS_X', 'IBS_PS_Y', 'IBS_SPS_X', 'IBS_SPS_Y']).transpose()

# Rename the columns for clarity
df_O_IBS.columns = [f'O_{col}' for col in df_O_IBS.columns]
df_Pb_IBS.columns = [f'Pb_{col}' for col in df_Pb_IBS.columns]
df_frac = df_O_IBS.divide(df_Pb_IBS.values).T
df_frac.index = index
df_frac.index.name = 'Frac O / Pb'

# Combine the dataframes for O, Pb, and the reference values
df_combined = pd.concat([df_frac , df_ref_frac], axis=1)

# Display the resulting dataframe
print(df_combined)
