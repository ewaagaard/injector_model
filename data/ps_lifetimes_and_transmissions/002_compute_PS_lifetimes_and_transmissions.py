"""
Main script to take calculated projectile input and compute PS lifetime from beam-gas interactions
"""
import pandas as pd
import beam_gas_collisions
import numpy as np

# Load pre-calculated projectile data
class DataObject:
    def __init__(self):
        self.projectile_data = pd.read_csv('PS_SPS_strip_all_projectile_data.csv', index_col=0)

# Load data 
projectile = 'Pb54'
data = DataObject()

# Instantiate classes for LEIR, PS, SPS
PS = beam_gas_collisions.IonLifetimes(machine='PS')
taus_PS_inj = np.zeros(len(data.projectile_data))
taus_PS_extr = np.zeros(len(data.projectile_data))

i = 0
for projectile, row in data.projectile_data.iterrows():
    
    # First calculate at injection energy
    PS.projectile = projectile
    PS.set_projectile_data(data)
    taus_PS_inj[i] = PS.calculate_total_lifetime_full_gas()
    
    # Then at PS extraction energy, same as SPS injection
    PS.gamma = row.SPS_gamma 
    PS.beta = row.SPS_beta
    PS.e_kin = row.SPS_Kinj
    taus_PS_extr[i] = PS.calculate_total_lifetime_full_gas()
    i += 1

df_tau = pd.DataFrame({'Projectile': data.projectile_data.index, 
                       'tau_PS_inj': taus_PS_inj,
                       'tau_PS_extr': taus_PS_extr,
                       'tau_PS_avg': (taus_PS_inj + taus_PS_extr)/2})
df_tau = df_tau.set_index(df_tau.columns[0])
df_tau.to_csv('PS_tau_values.csv')