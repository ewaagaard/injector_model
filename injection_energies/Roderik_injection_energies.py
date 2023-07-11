"""
Module to calculate and store injection energies for all ions, as used by Roderik in his study
"""
import pandas as pd
from injector_model import InjectorChain
from collections import defaultdict

# Load all the ion isotope data
ion_data = pd.read_csv("../data/Ion_species.csv", sep=';', header=0, index_col=0)
df = ion_data.T

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

##### Calculate gammas, E_kin per nucleon and Brho for all isotopes #### 
for i, row in ion_data.iterrows():
    
    ion_type = i
    print(ion_type)
    i1.init_ion(ion_type)

    result = i1.calculate_LHC_bunch_intensity()
    
    
    gamma_dict = {
        "Ion": i,
        "A": row['A'],
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
    #"PS_B_field_is_too_low": i1.PS_B_field_is_too_low
    #"p_LEIR_extr_proton_equiv": 1e-9 *i1.p_LEIR_extr_proton_equiv,
    #"p_PS_inj_proton_equiv": 1e-9 *i1.p_PS_inj_proton_equiv,
    #"p_PS_extr_proton_equiv": 1e-9 *i1.p_PS_extr_proton_equiv,
    #"p_SPS_inj_proton_equiv": 1e-9 *i1.p_SPS_inj_proton_equiv, 
    #"p_SPS_extr_proton_equiv": 1e-9 *i1.p_SPS_extr_proton_equiv,
    #"LEIR_Brho": i1.LEIR_Brho,
    #"PS_Brho_inj": i1.Brho_PS_inj,
    #"PS_Brho_extr": i1.Brho_PS_extr,
    #"SPS_Brho_inj": i1.Brho_SPS_inj,
    #"SPS_Brho_extr": i1.Brho_SPS_extr,
    
    all_data.append(gamma_dict)
    
# Convert to CSV
df = pd.DataFrame(all_data)
df.to_csv('../data/ion_injection_energies_Roderik_baseline_case_PS_SPS_strip.csv', index=False)
