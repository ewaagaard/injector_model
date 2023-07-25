#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General test class for Injector Class with full SC tune shif
"""
from injector_model import InjectorChain_full_SC
import pandas as pd
import numpy as np
import xpart as xp
import xtrack as xt

# Import data 
#ion_data = pd.read_csv("../data/Ion_species.csv", sep=';', header=0, index_col=0).T
ion_type = 'Pb'

# load data from Bartosik and John
dQ_data = pd.read_csv('../data/test_and_benchmark_data/John_and_Bartosik_tune_shifts.csv', header=0, index_col=0)


class TestClass_injectorModel_full_SC:
    """
    Test class for comparing SC tune shifts of full lattice integral,
    using Bartosik and John (2021) as reference (https://cds.cern.ch/record/2749453)
    """
    
    def test_detuning_Pb(self):        
        inj = InjectorChain_full_SC('Pb')
        dQ_Pb = np.array([inj.dQx0_LEIR, inj.dQy0_LEIR, inj.dQx0_PS, inj.dQy0_PS,inj.dQx0_SPS, inj.dQy0_SPS])
        
        assert np.all(np.isclose(dQ_Pb, dQ_data.loc['Pb'], rtol=5e-2))

    def test_detuning_O(self):
        """
        Compare SC detuning of O: custumized gamma and Nb from Bartosik and John paper 
        """
        inj = InjectorChain_full_SC('O')
           	
        ###### LEIR ######
        Nb_LEIR = 110e8
        gamma_LEIR = 1.00451
        sigma_z_LEIR = 8.5
        	
        particle_LEIR = xp.Particles(mass0 = 1e9 * inj.mass_GeV, q0 = inj.Q_LEIR, gamma0 = gamma_LEIR)
        line_LEIR_O = xt.Line.from_json('../data/xtrack_sequences/LEIR_2021_Pb_ions_with_RF.json')
        line_LEIR_O.reference_particle = particle_LEIR
        line_LEIR_O.build_tracker()
        twiss_LEIR = line_LEIR_O.twiss()
        twiss_LEIR_interpolated, sigma_x_LEIR, sigma_y_LEIR = inj.interpolate_Twiss_table(twiss_LEIR, 
                                                                                            line_LEIR_O, 
                                                                                            particle_LEIR, 
                                                                                            inj.ex_LEIR, 
                                                                                            inj.ey_LEIR,
                                                                                            inj.delta_LEIR,
                                                                                            )
        dQx_LEIR, dQy_LEIR = inj.calculate_SC_tuneshift(Nb_LEIR, particle_LEIR, sigma_z_LEIR, 
                         twiss_LEIR_interpolated, sigma_x_LEIR, sigma_y_LEIR)
        
        ###### PS ######
        Nb_PS = 88e8
        gamma_PS = 1.07214
        sigma_z_PS = 4.74
        	
        particle_PS = xp.Particles(mass0 = 1e9 * inj.mass_GeV, q0 = inj.Q_PS, gamma0 = gamma_PS)
        line_PS_O = xt.Line.from_json('../data/xtrack_sequences/PS_2022_Pb_ions_matched_with_RF.json')
        line_PS_O.reference_particle = particle_PS
        line_PS_O.build_tracker()
        twiss_PS = line_PS_O.twiss()
        twiss_PS_interpolated, sigma_x_PS, sigma_y_PS = inj.interpolate_Twiss_table(twiss_PS, 
                                                                                            line_PS_O, 
                                                                                            particle_PS, 
                                                                                            inj.ex_PS, 
                                                                                            inj.ey_PS,
                                                                                            inj.delta_PS,
                                                                                            )
        dQx_PS, dQy_PS = inj.calculate_SC_tuneshift(Nb_PS, particle_PS, sigma_z_PS, 
                         twiss_PS_interpolated, sigma_x_PS, sigma_y_PS)
        
        ###### SPS ######
        Nb_SPS = 50e8
        gamma_SPS = 7.04405
        sigma_z_SPS = 0.23
        	
        particle_SPS = xp.Particles(mass0 = 1e9 * inj.mass_GeV, q0 = inj.Q_SPS, gamma0 = gamma_SPS)
        line_SPS_O = xt.Line.from_json('../data/xtrack_sequences/SPS_2021_Pb_ions_matched_with_RF.json')
        line_SPS_O.reference_particle = particle_SPS
        line_SPS_O.build_tracker()
        twiss_SPS = line_SPS_O.twiss()
        twiss_SPS_interpolated, sigma_x_SPS, sigma_y_SPS = inj.interpolate_Twiss_table(twiss_SPS, 
                                                                                            line_SPS_O, 
                                                                                            particle_SPS, 
                                                                                            inj.ex_SPS, 
                                                                                            inj.ey_SPS,
                                                                                            inj.delta_SPS,
                                                                                            )
        dQx_SPS, dQy_SPS = inj.calculate_SC_tuneshift(Nb_SPS, particle_SPS, sigma_z_SPS, 
                         twiss_SPS_interpolated, sigma_x_SPS, sigma_y_SPS)
        
        # Compare with calculated tune shifts from their paper
        dQ_O = np.array([dQx_LEIR, dQy_LEIR, dQx_PS, dQy_PS, dQx_SPS, dQy_SPS])
        
        assert np.all(np.isclose(dQ_O, dQ_data.loc['O'], rtol=5e-2))
    
    
    def test_Nb_from_max_dQ(self):
        """
        Check if Nb from maximum tune shift agrees with input Nb to calculate tune shift
        --> test the formula backwards
        """    
        # First calculate Pb tune shifts from standard intensities
        inj = InjectorChain_full_SC('Pb')
        
        # Calculate max intensities for LEIR, PS and SPS 
        Nb_x_max_LEIR, Nb_y_max_LEIR = inj.maxIntensity_from_SC_integral(inj.dQx0_LEIR, inj.dQy0_LEIR, 
                                                                         inj.particle0_LEIR, inj.sigma_z_LEIR,
                                                                         inj.twiss0_LEIR_interpolated, inj.sigma_x0_LEIR, 
                                                                         inj.sigma_y0_LEIR
                                                                         ) 
        Nb_x_max_PS, Nb_y_max_PS = inj.maxIntensity_from_SC_integral(inj.dQx0_PS, inj.dQy0_PS, 
                                                                         inj.particle0_PS, inj.sigma_z_PS,
                                                                         inj.twiss0_PS_interpolated, inj.sigma_x0_PS, 
                                                                         inj.sigma_y0_PS
                                                                         ) 
        Nb_x_max_SPS, Nb_y_max_SPS = inj.maxIntensity_from_SC_integral(inj.dQx0_SPS, inj.dQy0_SPS, 
                                                                         inj.particle0_SPS, inj.sigma_z_SPS,
                                                                         inj.twiss0_SPS_interpolated, inj.sigma_x0_SPS, 
                                                                         inj.sigma_y0_SPS
                                                                         ) 
        ##### Compare the respective intensities - intensities should agree between x and y, and with input Nb for tune shift
        # LEIR
        assert np.isclose(Nb_x_max_LEIR, Nb_y_max_LEIR, rtol=1e-2)
        assert np.isclose(Nb_x_max_LEIR, inj.Nb0_LEIR, rtol=1e-2)
    
        # PS 
        assert np.isclose(Nb_x_max_PS, Nb_y_max_PS, rtol=1e-2)
        assert np.isclose(Nb_x_max_PS, inj.Nb0_PS, rtol=1e-2)
        
        # SPS 
        assert np.isclose(Nb_x_max_SPS, Nb_y_max_SPS, rtol=1e-2)
        assert np.isclose(Nb_x_max_SPS, inj.Nb0_SPS, rtol=1e-2)
    
    	
     
        

