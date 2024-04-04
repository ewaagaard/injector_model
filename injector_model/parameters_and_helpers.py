"""
Container for helper functions and parameter data classes
"""
import numpy as np
import xpart as xp
import xtrack as xt
from dataclasses import dataclass

@dataclass
class Inj_Parameters:
    """
    Data class to store CERN Ion Injector Chain parameters for nominal Pb

    If not measured, default values from Table (1) in John and Bartosik, 2021 (https://cds.cern.ch/record/2749453)
    All emittances are normalized
    """
    ### LEIR ###
    Nb0_LEIR = 1e9
    ex_LEIR = 0.4e-6
    ey_LEIR = 0.4e-6
    sigma_z_LEIR = 4.256
    delta_LEIR = 1.18e-3
    
    ### PS ###
    Nb0_PS = 8.1e8
    ex_PS = 0.8e-6
    ey_PS = 0.5e-6
    sigma_z_PS = 4.74
    delta_PS = 0.63e-3
    PS_MinB = 383 * 1e-4 # [T] - minimum magnetic field in PS, (Gauss to Tesla) from Heiko Damerau
    PS_MaxB = 1.26 # [T] - minimum magnetic field in PS, from reyes Alemany Fernandez
    PS_rho = 70.1206 # [m] - PS bending radius 

    ### SPS ###
    Nb0_SPS = 3.5e8
    ex_SPS = 1.3e-6
    ey_SPS = 0.9e-6
    sigma_z_SPS = 0.23
    delta_SPS = 1e-3

