"""
Module to calculate and store injection energies for all ions, as used by Roderik in his study
"""
import pandas as pd
from injector_model import InjectorChain, InjectionEnergies
from scipy import constants

save_data = True 

##### Load all the ion isotope data for the baseline study - as used by Roderik ###
ion_data = pd.read_csv("../data/Ion_species.csv", sep=';', header=0, index_col=0)
df = ion_data.T

##### Load full isotope data to compare correct Pb injection energies ####
df_full = pd.read_csv('../data/Full_isotope_data.csv', index_col=0)
Pb_data = df_full.loc['54Pb208']
A, Q_low, m_ion_in_u, Z = Pb_data['A'], Pb_data['Q_low'], Pb_data['m_ion_in_u'], Pb_data['Z']
inj_energies0 = InjectionEnergies(A, Q_low, m_ion_in_u, Z)

# Initialize injector chain 
i1 = InjectorChain('Pb', 
                    df, 
                    nPulsesLEIR = 0,
                    LEIR_bunches = 2,
                    PS_splitting = 2,
                    account_for_SPS_transmission=True,
                    consider_PS_space_charge_limit=False
                    )

# Make empty dictionary for the data
all_data = []
all_data_2 = []

##### Calculate gammas, E_kin per nucleon and Brho for all isotopes #### 
for i, row in ion_data.iterrows():
    
    ion_type = i
    print("\nRunning {}".format(ion_type))
    
    # Method 1: Roderik's way with fixed Brho at PS extraction 
    i1.init_ion(ion_type)
    result = i1.calculate_LHC_bunch_intensity()
    gamma_dict = {
        "Ion": i,
        "A": row['A'],
        "Q_low": row['Q before stripping'],
        "Z": row['Z'],
        "LEIR_gamma_inj": i1.gamma_LEIR_inj,
        "LEIR_gamma_extr": i1.gamma_LEIR_extr,
        "PS_gamma_inj": i1.gamma_PS_inj,
        "PS_gamma_extr": i1.gamma_PS_extr,
        "SPS_gamma_inj": i1.gamma_SPS_inj,
        "SPS_gamma_extr": i1.gamma_SPS_extr,
        "LEIR_Ekin_per_A_inj": i1.E_kin_per_A_LEIR_inj,
        "LEIR_Ekin_per_A_extr": i1.E_kin_per_A_LEIR_extr,
        "PS_Ekin_per_A_inj": i1.E_kin_per_A_PS_inj,
        "PS_Ekin_per_A_extr": i1.E_kin_per_A_PS_extr,
        "SPS_Ekin_per_A_inj": i1.E_kin_per_A_SPS_inj,
        "SPS_Ekin_per_A_extr": i1.E_kin_per_A_SPS_extr,
    }

    all_data.append(gamma_dict)
    
    # Method 2: matching Brho at every extraction and injection
    ion_mass_in_u = (1e9 * row['mass [GeV]']) / constants.physical_constants['atomic mass unit-electron volt relationship'][0]       
    inj_energies = InjectionEnergies(row['A'], row['Q before stripping'], ion_mass_in_u, row['Z'])
    inj_energies.PS_B = 1.24019039865  # update to PS extraction B-field used by Roderik 
    inj_energies.calculate_all_gammas()
    data_2 = inj_energies.return_all_gammas()
    
    gamma_dict_2 = {
        "Ion": i,
        "A": row['A'],
        "Q_low": row['Q before stripping'],
        "Z": row['Z'],
        **data_2,
    }    
    all_data_2.append(gamma_dict_2)
        
# Convert to CSV
df = pd.DataFrame(all_data)
df2 = pd.DataFrame(all_data_2)

print("\nTwo relevant comparison cases:\n")
print('\nComparing LEIR injection:\n')
print(df["LEIR_gamma_inj"]/df2["LEIR_gamma_inj"])
print('\nComparing SPS injection:\n')
print(df["SPS_gamma_inj"]/df2["SPS_gamma_inj"])

if save_data:
    df.to_csv('../data/injection_energies/test_ion_injection_energies_Roderik_baseline_case_PS_SPS_strip.csv', index=False)
    df2.to_csv('../data/injection_energies/test_ion_injection_energies_new_Brho_matching_model_baseline_case_PS_SPS_strip.csv', index=False)
