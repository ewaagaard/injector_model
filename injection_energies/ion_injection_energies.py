"""
Module to calculate and store injection energies for all ions
"""
import pandas as pd
from injector_model import InjectionEnergies

# New LEIR Brho
LEIR_brho_new = 6.7

# Load all the ion isotope data
df = pd.read_csv('../data/Full_isotope_data.csv', index_col=0)

##### Test Pb injection energies ####
Pb_data = df.loc['54Pb208']
A, Q_low, m_ion_in_u, Z = Pb_data['A'], Pb_data['Q_low'], Pb_data['m_ion_in_u'], Pb_data['Z']
inj_energies0 = InjectionEnergies(A, Q_low, m_ion_in_u, Z)

### Test with normal PS-SPS stripping
inj_energies0.calculate_all_gammas()
inj_energies0.print_all_gammas()

### Test with LEIR-PS stripping
inj_energies1 = InjectionEnergies(A, Q_low, m_ion_in_u, Z, LEIR_PS_strip=True)
inj_energies1.calculate_all_gammas()
inj_energies1.print_all_gammas()

### Test with normal PS-SPS stripping - higher Brho in LEIR
inj_energies2 = InjectionEnergies(A, Q_low, m_ion_in_u, Z)
inj_energies2.LEIR_Brho  = LEIR_brho_new
inj_energies2.calculate_all_gammas()
inj_energies2.print_all_gammas()

### Test with LEIR-PS stripping - higher Brho in LEIR
inj_energies3 = InjectionEnergies(A, Q_low, m_ion_in_u, Z, LEIR_PS_strip=True)
inj_energies3.LEIR_Brho  = LEIR_brho_new
inj_energies3.calculate_all_gammas()
inj_energies3.print_all_gammas()


# Initiate empty dictionaries
all_data = []
all_data_LEIR_PS = []
all_data_Brho = []
all_data_LEIR_PS_Brho = []

##### Calculate gammas, E_kin per nucleon and Brho for all isotopes #### 
for i, row in df.iterrows():

    # First run the case with normal PS-SPS stripping
    inj_energies = InjectionEnergies(row['A'], row['Q_low'], row['m_ion_in_u'], row['Z'])
    inj_energies.calculate_all_gammas()
    gamma_dict = inj_energies.return_all_gammas()

    data = {
        "Ion": i,
        "A": row['A'],
        "Q_low": row['Q_low'],
        "m_ion_in_u": row['m_ion_in_u'],
        "Z": row['Z'],
        **gamma_dict,
    }    
    all_data.append(data)

    # Then run LEIR-SPS stripping
    inj_energies_LEIR_PS = InjectionEnergies(row['A'], row['Q_low'], row['m_ion_in_u'], row['Z'], LEIR_PS_strip=True)
    inj_energies_LEIR_PS.calculate_all_gammas()
    gamma_dict_LEIR_PS = inj_energies_LEIR_PS.return_all_gammas()

    data_LEIR_PS = {
        "Ion": i,
        "A": row['A'],
        "Q_low": row['Q_low'],
        "m_ion_in_u": row['m_ion_in_u'],
        "Z": row['Z'],
        **gamma_dict_LEIR_PS,
    }    
    all_data_LEIR_PS.append(data_LEIR_PS)

    # 3rd: case with normal PS-SPS stripping and higher LEIR Brho
    inj_energies_Brho = InjectionEnergies(row['A'], row['Q_low'], row['m_ion_in_u'], row['Z'])
    inj_energies_Brho.LEIR_Brho  = LEIR_brho_new
    inj_energies_Brho.calculate_all_gammas()
    gamma_dict_Brho = inj_energies_Brho.return_all_gammas()

    data_Brho = {
        "Ion": i,
        "A": row['A'],
        "Q_low": row['Q_low'],
        "m_ion_in_u": row['m_ion_in_u'],
        "Z": row['Z'],
        **gamma_dict_Brho,
    }    
    all_data_Brho.append(data_Brho)

    # 4th case: run LEIR-SPS stripping with higher LEIR Brho
    inj_energies_LEIR_PS_Brho = InjectionEnergies(row['A'], row['Q_low'], row['m_ion_in_u'], row['Z'], LEIR_PS_strip=True)
    inj_energies_LEIR_PS_Brho.LEIR_Brho  = LEIR_brho_new
    inj_energies_LEIR_PS_Brho.calculate_all_gammas()
    gamma_dict_LEIR_PS_Brho = inj_energies_LEIR_PS_Brho.return_all_gammas()

    data_LEIR_PS_Brho = {
        "Ion": i,
        "A": row['A'],
        "Q_low": row['Q_low'],
        "m_ion_in_u": row['m_ion_in_u'],
        "Z": row['Z'],
        **gamma_dict_LEIR_PS_Brho,
    }    
    all_data_LEIR_PS_Brho.append(data_LEIR_PS_Brho)

# Convert all the data to a big dataframe
df = pd.DataFrame(all_data)
df.to_csv('../data/ion_injection_energies_PS_SPS_strip.csv', index=False)

df_LEIR_PS = pd.DataFrame(all_data_LEIR_PS)
df_LEIR_PS.to_csv('../data/ion_injection_energies_LEIR_PS_strip.csv', index=False)

df_Brho = pd.DataFrame(all_data_Brho)
df_Brho.to_csv('../data/ion_injection_energies_PS_SPS_strip_higher_brho_LEIR.csv', index=False)

df_LEIR_PS_Brho = pd.DataFrame(all_data_LEIR_PS_Brho)
df_LEIR_PS_Brho.to_csv('../data/ion_injection_energies_LEIR_PS_strip_higher_brho_LEIR.csv', index=False)

