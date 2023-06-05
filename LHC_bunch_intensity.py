#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LHC bunch intensity calculator for given ion from second part of Roderik's notebook'
- by Elias Waagaard 
"""
import pandas as pd
import numpy as np
from scipy.constants import e

class LHC_bunch_intensity:
    
    def __init__(self, rules, ion_type_ref='Pb'):
      
        # Also initiate reference values
        self.rules = rules
        self.use_Roderiks_gamma = True # for the SPS 
        self.init_ion()
        self.ion_type_ref = ion_type_ref
        self.ion0_referenceValues() 

    def init_ion(self):
        """
        Initialize ion species for a given type 
        """
        self.mass_GeV = self.rules["massGeV"]
        self.Z = self.rules["Z"]
        self.A = self.rules["A"]
        self.Q = self.rules["chargeState"]

    def beta(self, gamma):
        """
        Relativistic beta factor from gamma factor 
        """
        return np.sqrt(1 - 1/gamma**2)

    def ion0_referenceValues(self):
        
        # LEIR
        self.E_kin_per_A_LEIR_inj = 4.2e-3 # kinetic energy per nucleon in LEIR before RF capture, same for all species
        self.E_kin_per_A_LEIR_extr = 7.22e-2 # kinetic energy per nucleon in LEIR at exit, same for all species
        
        # PS
        self.E_kin_per_A_PS_inj = 7.22e-2 # GeV/nucleon according to LIU design report 
        self.E_kin_per_A_PS_extr = 5.9 # GeV/nucleon according to LIU design report 
        
        # SPS
        self.E_kin_per_A_SPS_inj = 5.9 # GeV/nucleon according to LIU design report 
        self.E_kin_per_A_SPS_extr = 176.4 # GeV/nucleon according to LIU design report 
        
        # As of now, reference data exists only for Pb54+
        if self.ion_type_ref == 'Pb':    
            
            # Pb ion values
            self.m0_GeV = 193.687 # rest mass in GeV for Pb reference case 
            self.Z0 = 82.0
            
            # LEIR - reference case for Pb54+ --> BEFORE stripping
            self.Nq0_LEIR_extr = 10e10  # number of observed charges extracted at LEIR
            self.Q0_LEIR = 54.0
            self.Nb0_LEIR_extr = self.Nq0_LEIR_extr/self.Q0_LEIR
            self.gamma0_LEIR_inj = (self.m0_GeV + self.E_kin_per_A_LEIR_inj * 208)/self.m0_GeV
            self.gamma0_LEIR_extr = (self.m0_GeV + self.E_kin_per_A_LEIR_extr * 208)/self.m0_GeV
            
            # PS - reference case for Pb54+ --> BEFORE stripping
            self.Nq0_PS_extr = 8e10  # number of observed charges extracted at PS for nominal beam
            self.Q0_PS = 54.0
            self.Nb0_PS_extr = self.Nq0_PS_extr/self.Q0_PS
            self.gamma0_PS_inj = (self.m0_GeV + self.E_kin_per_A_PS_inj * 208)/self.m0_GeV
            self.gamma0_PS_extr = (self.m0_GeV + self.E_kin_per_A_PS_extr * 208)/self.m0_GeV
            
            # SPS - reference case for Pb82+ --> AFTER stripping
            self.Nb0_SPS_extr = 2.21e8/0.62 # outgoing ions per bunch from SPS (2015 values), adjusted for 62% transmission
            self.Q0_SPS = 82.0
            self.Nq0_SPS_extr = self.Nb0_SPS_extr*self.Q0_SPS
            self.gamma0_SPS_inj = (self.m0_GeV + self.E_kin_per_A_SPS_inj * 208)/self.m0_GeV
            self.gamma0_SPS_extr = (self.m0_GeV + self.E_kin_per_A_SPS_extr * 208)/self.m0_GeV    
        
        # LEIR
        self.gamma_LEIR_inj = (self.mass_GeV + self.E_kin_per_A_LEIR_inj * self.A)/self.mass_GeV
        self.gamma_LEIR_extr = (self.mass_GeV + self.E_kin_per_A_LEIR_extr * self.A)/self.mass_GeV
        if self.use_Roderiks_gamma:
            self.gamma_SPS_inj =  self.gamma0_SPS_inj*(self.Q/54)/(self.mass_GeV/self.m0_GeV)
        else:
            self.gamma_SPS_inj = (self.mass_GeV + self.E_kin_per_A_SPS_inj * self.A)/self.mass_GeV
    


    def linearIntensityLimit(self, m, gamma, Nb_0, charge_0, m_0, gamma_0, fully_stripped=True):
        """
        Linear intensity limit for new ion species for given bunch intensity 
        Nb_0 and parameters gamma_0, charge0, m_0 from reference ion species - assuming
        that space charge stays constant, and that
        emittance and bunch length are constant for all ion species
        """
        # Specify if fully stripped ion or not
        if fully_stripped:
            charge = self.Z
        else:
            charge = self.Q
        beta_0 = self.beta(gamma_0)
        beta = self.beta(gamma)
        linearIntensityFactor = (m/m_0)*(charge_0/charge)**2*(beta/beta_0)*(gamma/gamma_0)**2   
        
        return Nb_0*linearIntensityFactor     
    
    
    def calc_ion_int(self):
        rules = self.rules
        currentLINAC3 = rules["currentLINAC3"]
        pulseLengthLinac3 = rules["pulseLengthLinac3"]
        chargeState = rules["chargeState"]
        nPulsesLEIR = rules.get("nPulsesLEIR", 0)
        LEIRinjEfficiency = rules.get("LEIRinjEfficiency", 1)
        LEIRtransmission = rules.get("LEIRtransmission", 1)
        LEIRbunches = rules.get("LEIRbunches", 1)
        LEIRPSstrip = rules.get("LEIRPSstripping", False)
        stripEfficiencyPS = rules.get("strippingEfficiencyPS", 1)
        PStransmission = rules.get("PStransmission", 1)
        PSsplitting = rules.get("PSsplitting", 1)
        PSSPSstrippingEfficiency = rules.get("PSSPSstrippingEfficiency", 1)
        SPStransmission = rules.get("SPStransmission", 1)
        SPSslipstackingTransm = rules.get("SPSslipstackingTransm", 1)
        currentLINAC3 = rules.get("currentLINAC3", 0)
        pulseLengthLinac3 = rules.get("pulseLengthLinac3", 0)
        Z = rules.get("Z", 0)
        species = rules.get("species", "")
        
        # Calculate ion transmission for LEIR 
        ionsPerPulseLeir = (currentLINAC3 * pulseLengthLinac3) / (chargeState * e)
        #spaceChargeLimitLEIR = 1.85 * 10**9 * Z * self.m0_GeV / (massGeV * (1 - (chargeState / 54)**2 * (1 - self.gamma0_LEIR_inj**2)))
        spaceChargeLimitLEIR = self.linearIntensityLimit(
                                               m = self.mass_GeV, 
                                               gamma = self.gamma_LEIR_extr,  
                                               Nb_0 = self.Nb0_LEIR_extr, 
                                               charge_0 = self.Q0_LEIR, # partially stripped charged state 
                                               m_0 = self.m0_GeV,  
                                               gamma_0 = self.gamma0_LEIR_extr,  # use gamma at extraction
                                               fully_stripped=False
                                               )
        
        totalIntLEIR = ionsPerPulseLeir*nPulsesLEIR*LEIRinjEfficiency
        #totalIntLEIR = ionsPerPulseLeir * min(7, spaceChargeLimitLEIR / (ionsPerPulseLeir * LEIRinjEfficiency)) * LEIRinjEfficiency ##3 THIS MIN FUNC IS WRONGLY INTERPRETED FROM MATHEMATICA...
        #print("LEIR: {}".format(spaceChargeLimitLEIR / (ionsPerPulseLeir * LEIRinjEfficiency) * LEIRinjEfficiency))
        #print(spaceChargeLimitLEIR, totalIntLEIR)
    
        LEIRPSstrip, stripEfficiencyPS = False, 1
        if "LEIRPSstripping" in rules:
            LEIRPSstrip = rules["LEIRPSstripping"]
        if "strippingEfficiencyPS" in rules:
            stripEfficiencyPS = rules["strippingEfficiencyPS"]
    
        ionsPerBunchExtractedLEIR = LEIRtransmission * min(totalIntLEIR, spaceChargeLimitLEIR) / LEIRbunches
        ionsPerBunchExtractedPS = ionsPerBunchExtractedLEIR *(stripEfficiencyPS if LEIRPSstrip else 1) * PStransmission / PSsplitting
        
        #print( (1 if Z == chargeState or LEIRPSstrip else PSSPSstrippingEfficiency))
        ionsPerBunchSPSinj = ionsPerBunchExtractedPS * (PSSPSstrippingEfficiency if Z == chargeState or LEIRPSstrip else 0.9)
        
        # Calculate ion transmission for SPS 
        #spaceChargeLimitSPS = (2.21 * 10**8 / 0.62) * Z * self.m0_GeV / (massGeV * (1 + ((Z if LEIRPSstrip else chargeState) / 54) / (massGeV / (self.m0_GeV / GeV)))**2 * (1 -self.gamma0_SPS_inj**2))
        spaceChargeLimitSPS = self.linearIntensityLimit(
                                               m = self.mass_GeV, 
                                               gamma = self.gamma_SPS_inj,  
                                               Nb_0 = self.Nb0_SPS_extr, 
                                               charge_0 = self.Q0_SPS, 
                                               m_0 = self.m0_GeV,  
                                               gamma_0 = self.gamma0_SPS_inj,  # use gamma at extraction
                                               fully_stripped=True
                                               )
    
        ionsPerBunchLHC = min(spaceChargeLimitSPS, ionsPerBunchSPSinj) * SPStransmission * SPSslipstackingTransm
    
        result = {
            "ionsPerPulseLinac3": ionsPerPulseLeir,
            "nPulsesLEIR": nPulsesLEIR,
            "gammaLEIR": self.gamma_LEIR_inj,
            "LEIRmaxInt": totalIntLEIR,
            "LEIRspaceChargeLim": spaceChargeLimitLEIR,
            "extractedLEIRperBunch": ionsPerBunchExtractedLEIR,
            "extractedPSperBunch": ionsPerBunchExtractedPS,
            "gammaInjSPS": self.gamma_SPS_inj,
            "SPSmaxIntPerBunch": ionsPerBunchSPSinj,
            "SPSspaceChargeLim": spaceChargeLimitSPS,
            "SPSaccIntensity": min(spaceChargeLimitSPS, ionsPerBunchSPSinj),
            "LHCionsPerBunch": ionsPerBunchLHC,
            "strippingEfficiencyLEIRPS": stripEfficiencyPS,
            "LHCChargesPerBunch": ionsPerBunchLHC * Z,
            "chargeBeforeStrip": chargeState,
            "atomicNumber": Z,
            "massNumber": rules.get("A", 0),
            "LEIRsplitting": LEIRbunches,
            "species": species
        }
    
        result.update({k: v for k, v in rules.items() if k in [
            "currentLINAC3", "pulseLengthLinac3", "LEIRtransmission",
            "LEIRinjEfficiency", "PSSPSstrippingEfficiency", "PStransmission",
            "PSsplitting", "SPStransmission", "SPSslipstackingTransm"
        ]})
    
        return result
    
# Test the class 
if __name__ == '__main__':
        
    rulesO4 = {
    "Z": 8,
    "A": 16,
    "chargeState": 4,
    "massGeV": 14.895092181928753,
    "currentLINAC3": 70 * 10**-6,
    "nPulsesLEIR": 1,
    "LEIRbunches": 1,
    "PSsplitting": 1,
    "SPSslipstackingTransm": 1,
    "LEIRPSstripping": False,
    "strippingEfficiencyPS": 0.6,
    "pulseLengthLinac3": 200 * 10**-6,
    "LEIRinjEfficiency": 0.5,
    "LEIRtransmission": 0.8,
    "PStransmission": 0.9,
    "PSSPSstrippingEfficiency": 0.9,
    "SPStransmission": 0.62
    }
    
    lhc = LHC_bunch_intensity(rulesO4)
    result = lhc.calc_ion_int()
    print("\nO4:")
    print(result)

#"""
    rules08 = {
        "Z": 8,
        "A": 16,
        "chargeState": 8,
        "massGeV": 14.895092181928753,
        "currentLINAC3": 70 * 10**-6,
        "nPulsesLEIR": 1,
        "LEIRbunches": 1,
        "PSsplitting": 1,
        "SPSslipstackingTransm": 1,
        "LEIRPSstripping": True,
        "strippingEfficiencyPS": 1,
        "pulseLengthLinac3": 200 * 10**-6,
        "LEIRinjEfficiency": 0.5,
        "LEIRtransmission": 0.8,
        "PStransmission": 0.9,
        "PSSPSstrippingEfficiency": 0.9,
        "SPStransmission": 0.62
    }
    
    lhc2 = LHC_bunch_intensity(rules08)
    result2 = lhc2.calc_ion_int()
    print("\nO8:")
    print(result2)
    
    rulesPb54 = {
        "Z": 82,
        "A": 208,
        "chargeState": 54,
        "massGeV": 193.687,
        "currentLINAC3": 70 * 10**-6,
        "nPulsesLEIR": 4,
        "LEIRbunches": 2,
        "PSsplitting": 1,
        "SPSslipstackingTransm": 1,
        "LEIRPSstripping": True,
        "strippingEfficiencyPS": 1,
        "pulseLengthLinac3": 200 * 10**-6,
        "LEIRinjEfficiency": 0.5,
        "LEIRtransmission": 0.8,
        "PStransmission": 0.9,
        "PSSPSstrippingEfficiency": 0.9,
        "SPStransmission": 0.62
    }
    
    lhc3 = LHC_bunch_intensity(rulesPb54)
    result3 = lhc3.calc_ion_int()
    print("\nPb54:")
    print(result3)
    
    # Test with slip-stacking loss: 
    rulesPb54["SPSslipstackingTransm"] = 1.9/2.2
    lhc4 = LHC_bunch_intensity(rulesPb54)
    result4 = lhc4.calc_ion_int()
    print("\nPb54 with slip-stacking loss:")
    print(result4)
    
    # Test with double PS splitting
    rulesPb54_PS_splitting = {
        "Z": 82,
        "A": 208,
        "chargeState": 54,
        "massGeV": 193.687,
        "currentLINAC3": 70 * 10**-6,
        "nPulsesLEIR": 6,
        "LEIRbunches": 2,
        "PSsplitting": 2,
        "SPSslipstackingTransm": 1,
        "LEIRPSstripping": True,
        "strippingEfficiencyPS": 1,
        "pulseLengthLinac3": 200 * 10**-6,
        "LEIRinjEfficiency": 0.5,
        "LEIRtransmission": 0.8,
        "PStransmission": 0.9,
        "PSSPSstrippingEfficiency": 0.9,
        "SPStransmission": 0.62
    }
    
    lhc5 = LHC_bunch_intensity(rulesPb54_PS_splitting)
    result5 = lhc5.calc_ion_int()
    print("\nPb54 with double PS splitting:")
    print(result5)
    
    