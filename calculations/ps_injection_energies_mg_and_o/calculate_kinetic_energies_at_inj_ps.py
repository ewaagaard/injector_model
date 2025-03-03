"""
Script to calculate kinetic energy per nucleon at injection
"""
from injector_model import InjectionEnergies

def bfield_to_kinetic(B, q_PS, inj_energy_object):
    "Compute kinetic energy per nucleon from magnetic field"
    Brho = B * inj_energy_object.PS_rho  # magnetic rigidity when magnets have ramped 
    p_PS_extr = inj_energy_object.calcMomentum(Brho, q_PS)
    p_PS_extr_proton_equiv = p_PS_extr / q_PS
    E_kin_per_u_PS_extr = inj_energy_object.calcKineticEnergyPerNucleon(p_PS_extr, q_PS)
    return E_kin_per_u_PS_extr


# Data for Mg7 and O4 - ['A', 'Q_low', 'm_ion_in_u', 'Z']
Mg7_data = [24,7,23.985041697,12]
O4_data = [16,4,15.99491461957,8]

inj_Ek_Mg7 = InjectionEnergies(*Mg7_data)
inj_Ek_O4 = InjectionEnergies(*O4_data)

### Test with normal PS-SPS stripping
print('---- Mg7+ ----\n')
inj_Ek_Mg7.calculate_all_gammas()
inj_Ek_Mg7.print_all_gammas()

print('\n---- O4+ ----\n')
inj_Ek_O4.calculate_all_gammas()
inj_Ek_O4.print_all_gammas()

# Compute energy at intermediate plateau and halfway through ramp
Bs = [0.06849412805548026*2, 0.72]

for B in Bs:
    E_kin_per_u_PS_extr_Mg = bfield_to_kinetic(B, 7.0, inj_Ek_Mg7)
    E_kin_per_u_PS_extr_O = bfield_to_kinetic(B, 4.0, inj_Ek_O4)

    print(f'\nKinetic energy per u for B = {B}:\n')
    print('Mg7: {:.4f} GeV/u'.format(1e-9*E_kin_per_u_PS_extr_Mg))
    print('O4: {:.4f} GeV/u'.format(1e-9*E_kin_per_u_PS_extr_O))

