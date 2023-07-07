"""
Module to calculate and print injection energies for Pb
"""
import pandas as pd
from injector_model import InjectionEnergies

# Load all the ion isotope data
df = pd.read_csv('../data/Full_isotope_data.csv', index_col=0)

##### Test Pb injection energies ####
Pb_data = df.loc['54Pb208']
A, Q_low, m_ion_in_u, Z = Pb_data['A'], Pb_data['Q_low'], Pb_data['m_ion_in_u'], Pb_data['Z']
inj_energies0 = InjectionEnergies(A, Q_low, m_ion_in_u, Z)

### Test with normal PS-SPS stripping
inj_energies0.calculate_all_gammas()
inj_energies0.print_all_gammas()

