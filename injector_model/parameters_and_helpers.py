"""
Container for helper functions and parameter data classes

Parameters based on operational Pb values
Reference values
- Pb ions per bunch from 2016 HL-LHC LIU targets: https://edms.cern.ch/ui/file/1420286/2/LIU-Ions_beam_parameter_table.pdf
- emittances and dp/p from https://cds.cern.ch/record/2749453 - Bartosik & John, 2021
"""
import numpy as np
import xpart as xp
import xtrack as xt
from pathlib import Path
from dataclasses import dataclass

data_folder = Path(__file__).resolve().parent.joinpath('../data').absolute()

@dataclass
class BeamParams_SPS:
    """Data Container for SPS Pb default beam parameters"""
    Nb : float = 3.5e8 # LIU 2016 values
    exn : float =  1.3e-6 #1.1e-6 measured in 2023, but for smaller Nb and then larger emittance blow-up
    eyn : float = 0.9e-6
    sigma_z : float = 0.23
    delta : float = 1e-3
    sigma_delta : float = 5e-4  # from Momentum_Spread class
    sigma_delta_Ca : float = 7.4e-4 # Ca has higher energy inj. energy, probably why its momentum spread is different

@dataclass
class BeamParams_PS:
    """Data Container for PS Pb default beam parameters"""
    Nb : float =  8.1e8 # LIU 2016 values
    exn : float = 0.8e-6
    eyn : float = 0.5e-6
    sigma_z : float = 4.74 # around 5.0 measured in 2023 at injection for LHC Pb beams
    delta : float = 0.63e-3
    sigma_delta : float = 6e-4  # from Momentum_Spread class
    sigma_delta_Ca : float = 5e-4 # Ca has higher energy inj. energy, probably why its momentum spread is different

@dataclass
class BeamParams_LEIR:
    """Data Container for LEIR Pb default beam parameters"""
    Nb : float = 19.1e8  # LIU 2016, corresponds to 10.3e10 charges
    exn : float = 0.4e-6
    eyn : float = 0.4e-6
    sigma_z : float = 8.0 # Isabelle had 4.256 m before, but seems to short
    delta: float = 1.18e-3
    Nb_isabelle : float = 1e9
    sigma_delta : float = 2.4e-3  # from Momentum_Spread class
    sigma_delta_Ca : float = 3e-3 # Ca has higher energy inj. energy, probably why its momentum spread is different

@dataclass
class Reference_Values:
    """
    Data class to store CERN Ion Injector Chain parameters for nominal Pb,
    including injection/extraction energies, number of charges extracted today, etc

    If not measured, default values from Table (1) in John and Bartosik, 2021 (https://cds.cern.ch/record/2749453)
    For injection and extraction energy, use known Pb ion values from LIU report on 
    https://edms.cern.ch/ui/file/1420286/2/LIU-Ions_beam_parameter_table.pdf
    """
    ##### Pb rest mass and final charge #####
    m0_GeV = 193.687 # rest mass in GeV for Pb reference case 
    A0 : int = 208  # mass number
    Z0 = 82.0  # atomic number

    ### LEIR reference case for Pb54+ --> BEFORE stripping ###
    max_injections_into_LEIR = 7
    E_kin_per_A_LEIR_inj = 4.2e-3 # kinetic energy per nucleon in LEIR before RF capture, same for all species
    E_kin_per_A_LEIR_extr = 7.22e-2 # kinetic energy per nucleon in LEIR at exit, same for all species
    Nq0_LEIR_extr = 10e10  # number of observed charges extracted at LEIR
    Q0_LEIR = 54.0

    ### PS reference case for Pb54+ --> BEFORE stripping ###
    E_kin_per_A_PS_inj = 7.22e-2 # GeV/nucleon according to LIU design report 
    E_kin_per_A_PS_extr = 5.9 # GeV/nucleon according to LIU design report 
    PS_MinB = 383 * 1e-4 # [T] - minimum magnetic field in PS, (Gauss to Tesla) from Heiko Damerau
    PS_MaxB = 1.26 # [T] - minimum magnetic field in PS, from reyes Alemany Fernandez
    PS_rho = 70.1206 # [m] - PS bending radius 
    Nq0_PS_extr =  6e10 # from November 2022 ionlifetime MD, previously 8e10  # number of observed charges extracted at PS for nominal beam
    Q0_PS = 54.0

    ### SPS reference case for Pb82+ --> AFTER stripping ###
    E_kin_per_A_SPS_inj = 5.9 # GeV/nucleon according to LIU design report 
    E_kin_per_A_SPS_extr = 176.4 # GeV/nucleon according to LIU design report 
    Q0_SPS = 82.0

    # General rules - stripping and transmission
    LEIR_injection_efficiency = 0.5
    LEIR_transmission = 0.85
    PS_transmission = 0.95
    PS_SPS_transmission_efficiency = 1.0 # 0.9 is what we see today with stripping, but Roderik uses 1.0 if we strip LEIR-PS
    PS_SPS_stripping_efficiency = 0.9  # default value until we have other value
    SPS_transmission = 0.55 # # old value 0.62, when old PS values and not new LIU 2016 parameters used. Discussed with Reyes 2024-03-18 from last year's performance, then
    # 0.79 reasonable, but then starting intensity Nb0 = 2.5e8 ions was used. For space charge limit, use Nb = 3.5e8 and 0.62 as transmission
    SPS_slipstacking_transmission = 1.0 # for now assume the 0.98 transmission is most likely included in SPS transmission value to match LIU ion values for LHC
            
    def __post_init__(self):
        ###### LEIR #####
        self.Nb0_LEIR_extr = self.Nq0_LEIR_extr/self.Q0_LEIR
        self.gamma0_LEIR_inj = (self.m0_GeV + self.E_kin_per_A_LEIR_inj * self.A0)/self.m0_GeV
        self.gamma0_LEIR_extr = (self.m0_GeV + self.E_kin_per_A_LEIR_extr * self.A0)/self.m0_GeV
        
        ##### PS #####
        self.Nb0_PS_extr = self.Nq0_PS_extr/self.Q0_PS
        self.gamma0_PS_inj = (self.m0_GeV + self.E_kin_per_A_PS_inj * self.A0)/self.m0_GeV
        self.gamma0_PS_extr = (self.m0_GeV + self.E_kin_per_A_PS_extr * self.A0)/self.m0_GeV
        
        ##### SPS #####
        self.Nb0_SPS_extr = 2.21e8/self.SPS_transmission # outgoing ions per bunch from SPS (2015 values), adjusted for 62% transmission
        self.Nq0_SPS_extr = self.Nb0_SPS_extr*self.Q0_SPS
        self.gamma0_SPS_inj = (self.m0_GeV + self.E_kin_per_A_SPS_inj * self.A0)/self.m0_GeV
        self.gamma0_SPS_extr = (self.m0_GeV + self.E_kin_per_A_SPS_extr * self.A0)/self.m0_GeV

