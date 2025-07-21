"""
Main class to generate projectile input file for PS lifetimes
"""
import pandas as pd
import numpy as np
from injector_model import InjectionEnergies

# Read projectile and full isotope data
ion_data = pd.read_csv("../Ion_species.csv", header=0, index_col=0).T
df_isotopes = pd.read_csv('../Full_isotope_data.csv', index_col=0)

# Which charge states to scan over, for standard isotopes
Q_states = {'He': [1, 2],
            'O': [4, 5, 6],
            'Mg': [7, 8],
            'Ar': [9, 10, 11],
            'Ca': np.arange(12, 21),
            'Kr': np.arange(19, 35),
            'In': np.arange(32, 43),
            'Xe': np.arange(34, 45),
            'Pb': np.arange(48, 59)}

LEIR_PS_strip_bools = [False, True]
LEIR_PS_strip_strings = ['PS_SPS_strip', 'LEIR_PS_strip']



print('\nPS SPS STRIP:')
projectile_dict = {'Projectile': [],
                   'Z_p' : [],
                   'A_p' : [],
                   'LEIR_Kinj' : [],
                   'PS_Kinj' : [],
                   'SPS_Kinj' : [],
                   'LEIR_beta' : [],
                   'PS_beta' : [],
                   'SPS_beta' : [],
                   'LEIR_gamma' : [],
                   'PS_gamma' : [],
                   'SPS_gamma' : [],
                   'LEIR_q' : [],
                   'PS_q' : [],
                   'SPS_q' : [],
                   'I_p': [],
                   'n_0': []
                   }


# Loop over all ions
for ion_type in ion_data:
    print('\n{}:'.format(ion_type))

    # Find values for specific ion
    A = ion_data[ion_type]['A']
    Z = ion_data[ion_type]['Z']
    Q_default = int(ion_data[ion_type]['Q before stripping'])
    m_ion_in_u = df_isotopes.loc['{}{}{}'.format(Q_default, ion_type, int(A))]['m_ion_in_u']
    
    # Read NIST data prepared by Roderik
    df_nist = pd.read_csv('nist_data/{}.csv'.format(ion_type), index_col=0)
    
    # Scan over all tested charge states
    for Q_low in Q_states[ion_type]:
        print('Q_low: {}+'.format(Q_low))
        Q_LEIR = Q_low
        Q_PS = Q_low # strip after the PS
        Q_SPS = Z # always fully stripped
        
        # Instantiate injection energy class, assume PS-SPS stripping
        inj = InjectionEnergies(A, Q_low, m_ion_in_u, Z, LEIR_PS_strip=False)
        gamma_dict = inj.return_all_gammas()
        
        # Append all the values
        projectile_dict['Projectile'].append('{}{}'.format(ion_type, Q_PS))
        projectile_dict['Z_p'].append(Z)
        projectile_dict['A_p'].append(A)
        projectile_dict['LEIR_Kinj'].append(1e3*gamma_dict['LEIR_Ekin_per_u_inj']) # convert GeV/u to MeV/u
        projectile_dict['PS_Kinj'].append(1e3*gamma_dict['PS_Ekin_per_u_inj']) # convert GeV/u to MeV/u
        projectile_dict['SPS_Kinj'].append(1e3*gamma_dict['SPS_Ekin_per_u_inj']) # convert GeV/u to MeV/u
        projectile_dict['LEIR_beta'].append(gamma_dict['LEIR_beta_inj'])
        projectile_dict['PS_beta'].append(gamma_dict['PS_beta_inj'])
        projectile_dict['SPS_beta'].append(gamma_dict['SPS_beta_inj'])
        projectile_dict['LEIR_gamma'].append(gamma_dict['LEIR_gamma_inj'])
        projectile_dict['PS_gamma'].append(gamma_dict['PS_gamma_inj'])
        projectile_dict['SPS_gamma'].append(gamma_dict['SPS_gamma_inj'])
        projectile_dict['LEIR_q'].append(Q_LEIR)
        projectile_dict['PS_q'].append(Q_PS)
        projectile_dict['SPS_q'].append(Q_SPS)
        projectile_dict['I_p'].append(1e-3*df_nist.loc[Q_PS]['Ip']) # convert to keV
        projectile_dict['n_0'].append(df_nist.loc[Q_PS]['n0'])
    
df_all = pd.DataFrame(projectile_dict)
df_all = df_all.set_index(df_all.columns[0])
df_all.to_csv('PS_SPS_strip_all_projectile_data.csv')