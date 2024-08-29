"""
Main class with methods to generate xtrack sequence with correct ion energy
"""
import numpy as np
import xpart as xp
import xtrack as xt
from pathlib import Path
from dataclasses import dataclass

data_folder = Path(__file__).resolve().parent.joinpath('../data').absolute()

class Sequences:
    """
    Generator class of sequences: LEIR, PS, SPS
    Provide m0_GeV (rest mass in GeV/c^2), charge state Q and relativistic gamma -> returns xt.Line
    """

    @staticmethod
    def get_SPS_line(m0_GeV, Q, gamma) -> xt.Line:
        """Returns SPS line with new ion as particle_ref"""
        line_SPS = xt.Line.from_json('{}/xtrack_sequences/SPS_2021_Pb_ions_matched_with_RF.json'.format(data_folder))
        line_SPS.particle_ref = xp.Particles(mass0 = 1e9 * m0_GeV, q0 = Q, gamma0 = gamma)
        line_SPS.build_tracker()
        return line_SPS
    
    @staticmethod
    def get_PS_line(m0_GeV, Q, gamma) -> xt.Line:
        """Returns PS line with new ion as particle_ref"""
        line_PS = xt.Line.from_json('{}/xtrack_sequences/PS_2022_Pb_ions_matched_with_RF.json'.format(data_folder))
        line_PS.particle_ref = xp.Particles(mass0 = 1e9 * m0_GeV, q0 = Q, gamma0 = gamma)
        line_PS.build_tracker()
        return line_PS

    @staticmethod
    def get_LEIR_line(m0_GeV, Q, gamma) -> xt.Line:
        """Returns LEIR line with new ion as particle_ref"""
        line_LEIR = xt.Line.from_json('{}/xtrack_sequences/LEIR_2021_Pb_ions_with_RF.json'.format(data_folder))
        line_LEIR.particle_ref = xp.Particles(mass0 = 1e9 * m0_GeV, q0 = Q, gamma0 = gamma)
        line_LEIR.build_tracker()
        return line_LEIR