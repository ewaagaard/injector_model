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


# Compare with Roderik's gamma for PS extraction - what B-field in the PS do we get? 
gamma_PS_extr = 7.33599
q_PS_extr = 54
p_PS_extr = inj_energies0.calcMomentum_from_gamma(gamma_PS_extr, q_PS_extr)
B_PS_extr = inj_energies0.calcBrho(p_PS_extr, q_PS_extr) / inj_energies0.PS_rho
print("\nRoderik's B-field at PS extraction: {}".format(B_PS_extr))