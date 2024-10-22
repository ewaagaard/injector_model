"""
Main space charge and IBS classes to calculate tune shifts and growth rates
"""
import numpy as np
import xpart as xp
import xtrack as xt
import pandas as pd
from dataclasses import dataclass
from .sequence_makers import Sequences
from .parameters_and_helpers import Reference_Values,BeamParams_LEIR, BeamParams_PS, BeamParams_SPS

# Import xibs for IBS growth rate calculations
from xibs.inputs import BeamParameters, OpticsParameters
from xibs.kicks import KineticKickIBS
from xibs.analytical import NagaitsevIBS
#from .parameters_and_helpers import Reference_Values, BeamParams_LEIR, BeamParams_PS, BeamParams_SPS

class SC_Tune_Shifts:
    """
    Class to analytically calculate space charge tune shift from 
    Eq. (1) in John and Bartosik, 2021 (https://cds.cern.ch/record/2749453)
    """
    def __init__(self) -> None:
        pass

    def beta(self, gamma):
        """
        Relativistic beta factor from gamma factor 
        """
        return np.sqrt(1 - 1/gamma**2)

    def interpolate_Twiss_table(self,
                                twissTableXsuite : xt.TwissTable,
                                particle_ref : xp.Particles,
                                line_length : float,
                                beamParams : dataclass,
                                interpolation_resolution = 1000000
                                ):
        """
        Interpolate Twiss table to higher resolution
        
        Parameters:
        -----------
        twissTableXsuite : xt.TwissTable
            twiss table from xtrack
        particle_ref : xp.Particles
            ion reference particle
        line_length : float
            length of sequence, can be retrieved with xt.Line.get_length()
        beamParams : dataclass
            dataclass containing exn, eyn, Nb, delta, sigma_delta

        Returns:
        --------
        twiss_xtrack_interpolated : xt.TwissTable
            interpolated twiss table with interpolation_resolution
        sigma_x : np.ndarray and sigma_y : np.ndarray
            array of horizontal and vertical beam sizes around lattice
        """
        gamma = particle_ref.gamma0[0]
        beta = self.beta(gamma)
        ss = np.linspace(0, line_length, interpolation_resolution)
        data2=np.zeros((interpolation_resolution, 8))
        data2[:,1] = np.square(np.interp(ss, twissTableXsuite['s'], np.sqrt(twissTableXsuite['betx'])))
        data2[:,2] = np.square(np.interp(ss, twissTableXsuite['s'], np.sqrt(twissTableXsuite['bety'])))
        data2[:,3] = np.interp(ss, twissTableXsuite['s'], beta * twissTableXsuite['dx'])
        data2[:,4] = np.interp(ss, twissTableXsuite['s'], beta * twissTableXsuite['dy'])
        data2[:,5] = np.interp(ss, twissTableXsuite['s'], twissTableXsuite['mux'])
        data2[:,6] = np.interp(ss, twissTableXsuite['s'], twissTableXsuite['muy'])
        data2[:,7] += line_length/len(ss)
        data2[:,0] = ss    
        data = data2
        columns = ['s', 'betx', 'bety', 'dx', 'dy', 'mux', 'muy', 'l']
        
        # Interpolate Twiss tables and beam sizes 
        twiss_xtrack_interpolated = pd.DataFrame(data, columns = columns)
        sigma_x = np.sqrt(beamParams.exn * twiss_xtrack_interpolated['betx'] / (beta * gamma) + (beamParams.delta * twiss_xtrack_interpolated['dx'])**2)
        sigma_y = np.sqrt(beamParams.eyn * twiss_xtrack_interpolated['bety'] / (beta * gamma) + (beamParams.delta * twiss_xtrack_interpolated['dy'])**2)
        
        return twiss_xtrack_interpolated, sigma_x, sigma_y
    

    def calculate_SC_tuneshift(self,
                               twissTableXsuite : xt.TwissTable,
                               particle_ref : xp.Particles,
                               line_length : float,
                               Nb : float,
                               beamParams : dataclass,
                               bF = 0.,
                               C = None,
                               h = None):
        """
        Finds the SC-induced max detuning dQx and dQy for given Twiss and beam parameters
        with possibility to use bunching factor bF, circumference C and harmonic h if non-Gaussian beams 

        Parameters:
        -----------
        twissTableXsuite : xt.TwissTable
            twiss table from xtrack
        particle_ref : xp.Particles
            ion reference particle
        line_length : float
            length of sequence, can be retrieved with xt.Line.get_length()
        Nb : float
            tentative bunch intensity of new ions
        beamParams : dataclass
            dataclass with beam parameters, assuming same exn, eyn, delta, sigma_delta, sigma_z for new ions. Contains Nb0 for Pb
        bF : float
            bunching factor - if zero, assume Gaussian shape
        C : float
            circumference of accelerator
        h : int
            harmonic of accelerator

        Returns:
        --------
        dQx, dQy : float
            Calculated analytical tune shift
        """  
        gamma = particle_ref.gamma0[0]
        beta = self.beta(gamma)
        r0 = particle_ref.get_classical_particle_radius0()
        
        # Calculated interpolated twiss table
        twiss_xtrack_interpolated, sigma_x, sigma_y = self.interpolate_Twiss_table(twissTableXsuite=twissTableXsuite, 
                                                                                   particle_ref=particle_ref,
                                                                                   line_length=line_length, beamParams=beamParams)

        # Space charge perveance
        if bF == 0.0:
            K_sc = (2 * r0 * Nb) / (np.sqrt(2*np.pi) * beamParams.sigma_z * beta**2 * gamma**3)
        else:
            K_sc = (2 * r0 * Nb) * (h / bF) / (C * beta**2 * gamma**3)

        # Numerically integrate SC lattice integral
        integrand_x = twiss_xtrack_interpolated['betx'] / (sigma_x * (sigma_x + sigma_y))  
        integrand_y = twiss_xtrack_interpolated['bety'] / (sigma_y * (sigma_x + sigma_y)) 
        
        dQx = - K_sc / (4 * np.pi) * np.trapz(integrand_x, x = twiss_xtrack_interpolated['s'])
        dQy = - K_sc / (4 * np.pi) * np.trapz(integrand_y, x = twiss_xtrack_interpolated['s'])
        
        return dQx, dQy


    def emittance_and_bunch_intensity_to_SC_tune_shift(self, exn, eyn, Nb, machine='SPS', particle_ref=None):
        """
        Compute space charge tune shifts for given emittances and bunch intensities
        
        Parameters:
        exn : float
            horizontal normalized emittance
        eyn : float 
            vertical normalized emittances
        Nb : float
            bunch intensity - ions per bunch
        machine : str
            which accelerator: LEIR, PS or SPS
        particle_ref : xp.Particles
            ion reference particle. Default is None
        """
        ref_val = Reference_Values()
        
        if machine == 'SPS':
            line = Sequences.get_SPS_line(m0_GeV = ref_val.m0_GeV, 
                                          Q = ref_val.Q0_SPS,
                                          gamma = ref_val.gamma0_SPS_inj)
            beamParams = BeamParams_SPS()
        elif machine == 'PS':
            line = Sequences.get_PS_line(m0_GeV = ref_val.m0_GeV, 
                                         Q = ref_val.Q0_PS, 
                                         gamma = ref_val.gamma0_PS_inj)
            beamParams = BeamParams_PS()
        elif machine == 'LEIR':
            line = Sequences.get_LEIR_line(m0_GeV = ref_val.m0_GeV, 
                                           Q = ref_val.Q0_LEIR, 
                                           gamma = ref_val.gamma0_LEIR_inj)
            beamParams = BeamParams_LEIR()
        else:
            raise ValueError('Machine not valid - either "LEIR", "PS" or "SPS"')
        
        # If particle_ref not provided, assume Pb default
        if particle_ref is None:
            particle_ref = line.particle_ref
            print('No ref. particle given - assume default Pb\n')
        print('\n' + particle_ref.show() + '\n')

        # Define new emittances
        beamParams.exn = exn
        beamParams.eyn = eyn
        beamParams.Nb = Nb
        print(beamParams)            
        
        # Calculate tune shifts
        dQx, dQy = self.calculate_SC_tuneshift(twissTableXsuite=line.twiss(),
                                                    particle_ref=particle_ref,
                                                    line_length=line.get_length(),
                                                    Nb=Nb,
                                                    beamParams=beamParams)
        return dQx, dQy


    def maxIntensity_from_SC_integral(self, 
                                      dQx_max : float, 
                                      dQy_max : float, 
                                      twissTableXsuite : xt.TwissTable,
                                      particle_ref : xp.Particles,
                                      line_length : float,
                                      beamParams : dataclass
                                      ):
        """
        For a given max tuneshift, calculate the maximum bunch intensity 

        Parameters:
        -----------
        dQx, dQy : float
            Calculated analytical tune shifts in X and Y
        twissTableXsuite : xt.TwissTable
            twiss table from xtrack
        particle_ref : xp.Particles
            ion reference particle
        line_length : float
            length of sequence, can be retrieved with xt.Line.get_length()
        beamParams : dataclass
            dataclass containing exn, eyn, delta, sigma_delta, sigma_z (assuming same for all ions), and Nb0 for Pb ions

        Returns:
        --------
        Nb_max : float
            maximum intensity for a given tune shift, min(Nb_x_max, Nb_y_max)         
        """
        gamma = particle_ref.gamma0[0]
        beta = self.beta(gamma)
        r0 = particle_ref.get_classical_particle_radius0()
        print('max intensity calculations: gamma = {:.3f}, r0 = {:.3e}'.format(gamma, r0)) # --> seem correct! 

        # Calculated interpolated twiss table
        twiss_xtrack_interpolated, sigma_x, sigma_y = self.interpolate_Twiss_table(twissTableXsuite=twissTableXsuite, 
                                                                                   particle_ref=particle_ref,
                                                                                   line_length=line_length, beamParams=beamParams)

        # Load interpolated beam sizes and Twiss parameters, then define SC integrands 
        integrand_x = twiss_xtrack_interpolated['betx'] / (sigma_x * (sigma_x + sigma_y))  
        integrand_y = twiss_xtrack_interpolated['bety'] / (sigma_y * (sigma_x + sigma_y)) 

        # Calculate maximum bunch intensity for given max tune shift in x and in y
        Nb_x_max = -dQx_max / (r0 * np.trapz(integrand_x, x = twiss_xtrack_interpolated['s'])) \
                    * (2 * np.pi * np.sqrt(2*np.pi) * beamParams.sigma_z * beta**2 * gamma**3)
        
        Nb_y_max = -dQy_max / (r0 * np.trapz(integrand_y, x = twiss_xtrack_interpolated['s'])) \
                    * (2 * np.pi * np.sqrt(2*np.pi) * beamParams.sigma_z * beta**2 * gamma**3)
        Nb_max = min(Nb_x_max, Nb_y_max)
        #print('\nNb_max = {:.3e}, between Nb_max_X: {:.3e} and Nb_max_Y: {:.3e}'.format(Nb_max, Nb_x_max, Nb_y_max))

        return Nb_max


class IBS_Growth_Rates:
    """
    Class to calculate IBS growth rates for a given lattice for given lattice and beam parameters
    """
    def __init__(self) -> None:
        pass

    def get_growth_rates(self, line, beamParams, also_calculate_kinetic=False, num_part=10_000):
        """
        Calculate analytical Nagaitsev IBS growth rates and kinetic growth rates
        from line and beam parameters 
        - kinetic kick assumes Gaussian bunch for now

        Parameters:
        -----------
        line : xt.Line
            xtrack line to use
        beamParams : dataclass
            beamParams class containing bunch intensity Nb, normalized emittances exn and eyn,
            sigma_delta and sigma_z (bunch_length)
        also_calculate_kinetic : bool
            whether to also calculate kinetic growth rates
        num_part : int
            number of macroparticles for xp.Part object for kinetic kick calculation
            
        Returns:
        --------
        growth_rates : np.ndarray
            array containing Tx, Ty and Tz - growth rates in respective plane
        """

        beamparams = BeamParameters.from_line(line, n_part=beamParams.Nb)
        opticsparams = OpticsParameters.from_line(line)

        # Instantiate analytical Nagaitsev IBS class
        NIBS = NagaitsevIBS(beamparams, opticsparams)
        growth_rates_in_class = NIBS.growth_rates(beamParams.exn, 
                                         beamParams.eyn, 
                                         beamParams.sigma_delta, 
                                         beamParams.sigma_z,
                                         normalized_emittances=True)
        growth_rates = np.array([growth_rates_in_class.Tx, growth_rates_in_class.Ty, growth_rates_in_class.Tz])
        
        if also_calculate_kinetic:
            # Instantiate kinetic IBS class
            IBS = KineticKickIBS(beamparams, opticsparams)

            # Generate Gaussian particle object 
            particles = xp.generate_matched_gaussian_bunch(
                num_particles=num_part, 
                total_intensity_particles=beamParams.Nb,
                nemitt_x=beamParams.exn, 
                nemitt_y=beamParams.eyn, 
                sigma_z= beamParams.sigma_z,
                particle_ref=line.particle_ref, 
                line=line)
            kinetic_kick_coefficients = IBS.compute_kick_coefficients(particles)

            return growth_rates, kinetic_kick_coefficients
        else:       
            return growth_rates

        