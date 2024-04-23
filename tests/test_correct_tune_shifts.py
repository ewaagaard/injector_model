"""
Test whether SC tune shift in LEIR, PS and SPS are what we expect - scale from Pb
"""
from injector_model import InjectorChain_v2

# Instantiate injector chain version 2 and calculate LHC bunch intensity for all ions
inj = InjectorChain_v2(nPulsesLEIR=7)  # 7 OR DEFAULT NUMBER?

for i, ion_type in enumerate(inj.full_ion_data.columns):
    
    # Initiate the correct ion, and calculate SC limit and tune shifts
    inj.init_ion(ion_type)
    Nb_spaceChargeLimitLEIR, dQx_LEIR, dQy_LEIR = inj.LEIR_SC_limit()
    Nb_spaceChargeLimitPS, dQx_PS, dQy_PS = inj.PS_SC_limit()
    Nb_spaceChargeLimitSPS, dQx_SPS, dQy_SPS = inj.SPS_SC_limit()
    print(f'Ion type: {ion_type}')
    print('LEIR: SC_limit = {:.3e}, dQx = {:.3f}, dQy = {:.3f}'.format(Nb_spaceChargeLimitLEIR, dQx_LEIR, dQy_LEIR))
    print('PS:   SC_limit = {:.3e}, dQx = {:.3f}, dQy = {:.3f}'.format(Nb_spaceChargeLimitPS, dQx_PS, dQy_PS))
    print('SPS:  SC_limit = {:.3e}, dQx = {:.3f}, dQy = {:.3f}\n'.format(Nb_spaceChargeLimitSPS, dQx_SPS, dQy_SPS))