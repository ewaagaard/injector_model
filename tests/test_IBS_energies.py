#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small test script to ensure that IBS module takes correct energies 
"""
import numpy as np
import xtrack as xt
import xpart as xp
from injector_model import InjectorChain_full_SC 
from lib.IBSfunctions import NagaitsevIBS

# Instantiate the injector chain object 
injectors = InjectorChain_full_SC('Pb')
injectors.simulate_injection()

def test_energy_for_IBS(gamma):
    
    #### SPS ####
    particle_SPS = xp.Particles(mass0 = 1e9 * injectors.mass_GeV, q0 = injectors.Q_SPS, gamma0 = gamma)
    line_SPS = injectors.line_SPS_Pb.copy()
    line_SPS.reference_particle = particle_SPS
    line_SPS.build_tracker()
    twiss_SPS = line_SPS.twiss()
    
    # Set up starting beam parameters
    n_part = 5000
    ex = injectors.ex_SPS
    ey = injectors.ey_SPS
    Nb = injectors.Nb0_SPS
    sigma_z = injectors.sigma_z_SPS
    
    #"""
    # Create Gaussian bunch of particle object
    particles = xp.generate_matched_gaussian_bunch(
                                                    num_particles = n_part, total_intensity_particles = Nb,
                                                    nemitt_x = ex, nemitt_y = ey, sigma_z = sigma_z,
                                                    particle_ref = particle_SPS, line = line_SPS
                                                    )
    
    # ----- Initialize IBS object -----
    IBS = NagaitsevIBS()
    IBS.set_beam_parameters(particles)
    IBS.set_optic_functions(twiss_SPS)
    print("IBS enTOT: {}".format(IBS.EnTot))       
    
    # --- Initialize first turn-by-turn data for all modes 
    sig_x = np.std(particles.x[particles.state > 0])
    sig_y = np.std(particles.y[particles.state > 0])
    sig_delta = np.std(particles.delta[particles.state > 0])
    bl = np.std(particles.zeta[particles.state > 0])
    eps_x = (sig_x**2 - (twiss_SPS['dx'][0] * sig_delta)**2) / twiss_SPS['betx'][0]
    eps_y = sig_y**2 / twiss_SPS['bety'][0] 
    
    # Calculate the analytical growth rates using the particle distribution
    IBS.calculate_integrals(
        eps_x,
        eps_y,
        sig_delta,
        bl
        )
    
    Ixx_SPS, Iyy_SPS, Ipp_SPS = injectors.calculate_IBS_growth_rate_for_SPS(Nb, gamma, sigma_z = sigma_z)
    print('\nParticles object parameters: eps_x: {}, eps_y: {}, sig_delta: {}, bl: {}'.format(eps_x, eps_y, sig_delta, bl))
    print('Input parameters:              eps_x: {}, eps_y: {}, sig_delta: {}, bl: {}'.format(ex, ey, injectors.delta_SPS, sigma_z))
    
    # Then check internal method from class 
    print("\n\nSPS gamma = {}".format(gamma))
    print("IBS integrals with analytical input:     {}, {}, {}\n\n".format(Ixx_SPS, Iyy_SPS, Ipp_SPS))
    print("IBS integrals with input from particles: {}, {}, {}\n\n".format(IBS.Ixx, IBS.Iyy, IBS.Ipp))
 

if __name__ == '__main__':
    
    # Test IBS module for some different energies 
    gamma_range = injectors.gamma0_SPS_inj * np.arange(1, 4)
    IBS_vals = np.zeros([len(gamma_range), 3])
    
    # loop over different energies 
    for i, gamma in enumerate(gamma_range):
        IBS_vals[i, :] = test_energy_for_IBS(gamma)
    #print("\n\nFinal IBS values: {}".format(IBS_vals))