"""
Container for class to calculate momentum spread
"""
import numpy as np
import xpart as xp
from .injector_model import InjectorChain
from .parameters_and_helpers import BeamParams_LEIR, BeamParams_PS, BeamParams_SPS

class Momentum_Spread(InjectorChain):
    """"Beam parameter class to calculate typical momentum spread values"""

    def __init__(self):
        InjectorChain.__init__(self)

    def calculate_sigma_delta_LEIR_for_all_ions(self, num_part=5000)->None:
        """Find momentum spread for all ions in LEIR"""
        leir_line = self.line_LEIR_Pb0.copy()
        leir_line.build_tracker()
        
        print('\nLEIR: Calculating momentum spread...')
        
        # Initialize empty array for all momentum spread
        sigma_deltas = np.zeros(len(self.full_ion_data.columns))
        
        # Iterate over all ion types
        for i, ion_type in enumerate(self.full_ion_data.columns):
            self.init_ion(ion_type)
            
            LEIR_particle = xp.Particles(mass0 = 1e9 * self.mass_GeV, q0 = self.Q_LEIR, gamma0 = self.LEIR_gamma_inj)
            leir_line.particle_ref = LEIR_particle
            
            particles_leir = xp.generate_matched_gaussian_bunch(
                num_particles=num_part, 
                total_intensity_particles=BeamParams_LEIR.Nb,
                nemitt_x=BeamParams_LEIR.exn, 
                nemitt_y=BeamParams_LEIR.eyn, 
                sigma_z= BeamParams_LEIR.sigma_z,
                particle_ref=leir_line.particle_ref, 
                line=leir_line)
            sigma_delta = np.std(particles_leir.delta[particles_leir.state > 0])
            sigma_deltas[i] = sigma_delta
            
            print('LEIR: for {}, sigma_delta = {}'.format(ion_type, sigma_delta))
        return sigma_deltas
            
            
    def calculate_sigma_delta_PS_for_all_ions(self, num_part=5000)->None:
        """Find momentum spread for all ions in PS"""
        ps_line = self.line_PS_Pb0.copy()
        ps_line.build_tracker()

        # Initialize empty array for all momentum spread
        sigma_deltas = np.zeros(len(self.full_ion_data.columns))

        print('\nPS: Calculating momentum spread...')
        # Iterate over all ion types
        for i, ion_type in enumerate(self.full_ion_data.columns):
            self.init_ion(ion_type)
            
            PS_particle = xp.Particles(mass0 = 1e9 * self.mass_GeV, q0 = self.Q_PS, gamma0 = self.PS_gamma_inj)   
            ps_line.particle_ref = PS_particle
            
            particles_ps = xp.generate_matched_gaussian_bunch(
                num_particles=num_part, 
                total_intensity_particles=BeamParams_PS.Nb,
                nemitt_x=BeamParams_PS.exn, 
                nemitt_y=BeamParams_PS.eyn, 
                sigma_z= BeamParams_PS.sigma_z,
                particle_ref=ps_line.particle_ref, 
                line=ps_line)
            sigma_delta = np.std(particles_ps.delta[particles_ps.state > 0])
            sigma_deltas[i] = sigma_delta
            
            print('PS: for {}, sigma_delta = {}'.format(ion_type, sigma_delta))
        return sigma_deltas


    def calculate_sigma_delta_SPS_for_all_ions(self, num_part=5000)->None:
        """Find momentum spread for all ions in SPS"""
        sps_line = self.line_SPS_Pb0.copy()
        sps_line.build_tracker()

        # Initialize empty array for all momentum spread
        sigma_deltas = np.zeros(len(self.full_ion_data.columns))

        print('\nSPS: Calculating momentum spread...')
        # Iterate over all ion types
        for i, ion_type in enumerate(self.full_ion_data.columns):
            self.init_ion(ion_type)
            
            SPS_particle = xp.Particles(mass0 = 1e9 * self.mass_GeV, q0 = self.Q_SPS, gamma0 = self.SPS_gamma_inj)
            sps_line.particle_ref = SPS_particle
            
            particles_sps = xp.generate_matched_gaussian_bunch(
                num_particles=num_part, 
                total_intensity_particles=BeamParams_SPS.Nb,
                nemitt_x=BeamParams_SPS.exn, 
                nemitt_y=BeamParams_SPS.eyn, 
                sigma_z= BeamParams_SPS.sigma_z,
                particle_ref=sps_line.particle_ref, 
                line=sps_line)
            sigma_delta = np.std(particles_sps.delta[particles_sps.state > 0])
            sigma_deltas[i] = sigma_delta
            
            print('SPS: for {}, sigma_delta = {}'.format(ion_type, sigma_delta))
        return sigma_deltas