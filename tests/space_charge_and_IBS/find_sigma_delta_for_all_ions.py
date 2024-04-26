"""
Test script to calculate typical momentum spread values (sigma delta) in each machine
"""
from injector_model import Momentum_Spread
import os
import pandas as pd

# Instantiate class and run calculations
delta = Momentum_Spread()
sig_delta_leir = delta.calculate_sigma_delta_LEIR_for_all_ions()
sig_delta_ps = delta.calculate_sigma_delta_PS_for_all_ions()
sig_delta_sps = delta.calculate_sigma_delta_SPS_for_all_ions()

print('\nIons: {}'.format(delta.full_ion_data.columns))
print('LEIR: {}'.format(sig_delta_leir))
print('PS: {}'.format(sig_delta_ps))
print('SPS: {}'.format(sig_delta_sps))

# Save data to csv file
os.makedirs('output_IBS', exist_ok=True)
T_dict = {'Ion':delta.full_ion_data.columns, 'LEIR': sig_delta_leir, 'PS':sig_delta_ps, 'SPS':sig_delta_sps}
df = pd.DataFrame(T_dict)
df.set_index('Ion')
df.to_csv("output_IBS/momentum_spread.csv")