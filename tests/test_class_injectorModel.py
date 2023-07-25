#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General test class for Injector Class to check
- incoming bunch intensity to the LHC 
"""
from injector_model import InjectorChain
import pandas as pd
import numpy as np

# Import data 
ion_data = pd.read_csv("../data/Ion_species.csv", sep=';', header=0, index_col=0).T
ion_type = 'Pb'

# Load reference values from Roderik Bruce's Mathematica notebook output (2021)
Roderik_LHC_charges_per_bunch = pd.read_csv('../data/Roderik_2021_LHC_charges_per_bunch_output.csv', index_col=0)
ref_val = Roderik_LHC_charges_per_bunch.sort_values('Z')


class TestClass_injectorModel_linearSpaceCharge:
    """
    Test class for comparing Injector Model with linear space charge (LEIR and SPS, not yet considering PS)
    versus estimates from Bruce (2021) values
    """
    
    def test_LHC_charges_vs_Bruce_baseline(self):        
        ### TEST CASE 1: BASELINE ###
        injector_chain = InjectorChain(ion_type, 
                                        ion_data, 
                                        nPulsesLEIR = 0,
                                        LEIR_bunches = 2,
                                        PS_splitting = 2,
                                        account_for_SPS_transmission=True,
                                        consider_PS_space_charge_limit=False
                                        )
        df_baseline = injector_chain.calculate_LHC_bunch_intensity_all_ion_species()
        assert np.all(np.isclose(ref_val['Baseline'].values, df_baseline['LHC_chargesPerBunch'].values, rtol=1e-2))
     
        
    def test_LHC_charges_vs_Bruce_no_PS_split(self):   
        ### TEST CASE 2: NO PS SPLITTING ###
        injector_chain2 = InjectorChain(ion_type, 
                                        ion_data, 
                                        nPulsesLEIR = 0,
                                        LEIR_bunches = 2,
                                        PS_splitting = 1,
                                        account_for_SPS_transmission=True,
                                        consider_PS_space_charge_limit=False
                                        )
        df_no_ps_split = injector_chain2.calculate_LHC_bunch_intensity_all_ion_species()
        assert np.all(np.isclose(ref_val['No_PS_splitting'].values, df_no_ps_split['LHC_chargesPerBunch'].values, rtol=1e-2))    
        
        
    def test_LHC_charges_vs_Bruce_LEIR_PS_stripping(self):   
        ### TEST CASE 3: PS SPLITTING AND LEIR-PS STRIPPING ###
        injector_chain3 = InjectorChain(ion_type, 
                                        ion_data, 
                                        nPulsesLEIR = 0,
                                        LEIR_bunches = 2,
                                        PS_splitting = 2,
                                        account_for_SPS_transmission=True,
                                        LEIR_PS_strip=True,
                                        consider_PS_space_charge_limit=False
                                        )
        df_ps_leir_strip = injector_chain3.calculate_LHC_bunch_intensity_all_ion_species()
        assert np.all(np.isclose(ref_val['LEIR_PS_strip'].values, df_ps_leir_strip['LHC_chargesPerBunch'].values, rtol=1e-2))   
        
        
    def test_LHC_charges_vs_Bruce_no_PS_splitting_and_LEIR_PS_stripping(self):   
        ### TEST CASE 4: NO SPLITTING AND LEIR-PS STRIPPING ###
        injector_chain4 = InjectorChain(ion_type, 
                                        ion_data, 
                                        nPulsesLEIR = 0,
                                        LEIR_bunches = 2,
                                        PS_splitting = 1,
                                        account_for_SPS_transmission=True,
                                        LEIR_PS_strip=True,
                                        consider_PS_space_charge_limit=False
                                        )
        df_no_ps_splitting_and_ps_leir_strip = injector_chain4.calculate_LHC_bunch_intensity_all_ion_species()
        assert np.all(np.isclose(ref_val['No_PS_split_and_LEIR_PS_strip'].values, 
                          df_no_ps_splitting_and_ps_leir_strip['LHC_chargesPerBunch'].values, rtol=1e-2))   
