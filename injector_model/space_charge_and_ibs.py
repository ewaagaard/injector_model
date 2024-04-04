"""
Main space charge and IBS classes to calculate tune shifts and growth rates
"""
import numpy as np
import xpart as xp
import xtrack as xt
import pandas as pd
from dataclasses import dataclass

# Import xibs for IBS growth rate calculations
from xibs.inputs import BeamParameters, OpticsParameters
from xibs.kicks import KineticKickIBS
from xibs.analytical import NagaitsevIBS

class SC_Tune_Shifts:
    """
    Class to analytically calculate space charge tune shift from 
    Eq. (1) in John and Bartosik, 2021 (https://cds.cern.ch/record/2749453)
    """
    
    def beta(self, gamma):
        """
        Relativistic beta factor from gamma factor 
        """
        return np.sqrt(1 - 1/gamma**2)

    def interpolate_Twiss_table(self, 
                                twissTableXsuite : xt.TwissTable,
                                line : xt.Line,
                                beamParams : dataclass,
                                interpolation_resolution = 1000000
                                ):
        """
        Interpolate Twiss table to higher resolution
        
        Parameters:
        -----------
        twissTableXsuite : xt.TwissTable
            Twiss table from xt.Line.twiss()
        beamParams : dataclass
            dataclass containing exn, eyn, Nb, delta, sigma_delta

        
        Xtrack twiss table and beam sizes sigma_x and sigma_y
        """
        gamma = line.particle_ref.gamma0[0]
        beta = self.beta(gamma)
        ss = np.linspace(0, line.get_length(), interpolation_resolution)
        data2=np.zeros((interpolation_resolution, 8))
        data2[:,1] = np.square(np.interp(ss, twissTableXsuite['s'], np.sqrt(twissTableXsuite['betx'])))
        data2[:,2] = np.square(np.interp(ss, twissTableXsuite['s'], np.sqrt(twissTableXsuite['bety'])))
        data2[:,3] = np.interp(ss, twissTableXsuite['s'], beta * twissTableXsuite['dx'])
        data2[:,4] = np.interp(ss, twissTableXsuite['s'], beta * twissTableXsuite['dy'])
        data2[:,5] = np.interp(ss, twissTableXsuite['s'], twissTableXsuite['mux'])
        data2[:,6] = np.interp(ss, twissTableXsuite['s'], twissTableXsuite['muy'])
        data2[:,7] += line.get_length()/len(ss)
        data2[:,0] = ss    
        data = data2
        columns = ['s', 'betx', 'bety', 'dx', 'dy', 'mux', 'muy', 'l']
        
        # Interpolate Twiss tables and beam sizes 
        twiss_xtrack_interpolated = pd.DataFrame(data, columns = columns)
        sigma_x = np.sqrt(beamParams.exn * twiss_xtrack_interpolated['betx'] / (beta * gamma) + (beamParams.delta * twiss_xtrack_interpolated['dx'])**2)
        sigma_y = np.sqrt(beamParams.eyn * twiss_xtrack_interpolated['bety'] / (beta * gamma) + (beamParams.delta * twiss_xtrack_interpolated['dy'])**2)
        
        return twiss_xtrack_interpolated, sigma_x, sigma_y
    

    def calculate_SC_tuneshift(self, 
                               Nb, 
                               particle_ref, 
                               sigma_z, 
                               twiss_xtrack_interpolated, 
                               sigma_x, sigma_y, 
                               bF = 0.,
                               C = None,
                               h = None):
        """
        Finds the SC-induced max detuning dQx and dQy for given Twiss and beam parameters
        with possibility to use bunching factor bF, circumference C and harmonic h if non-Gaussian beams 
        """  
        gamma = particle_ref.gamma0[0]
        beta = self.beta(gamma)
        r0 = particle_ref.get_classical_particle_radius0()
        
        # Space charge perveance
        if bF == 0.0:
            K_sc = (2 * r0 * Nb) / (np.sqrt(2*np.pi) * sigma_z * beta**2 * gamma**3)
        else:
            K_sc = (2 * r0 * Nb) * (h / bF) / (C * beta**2 * gamma**3)

        integrand_x = twiss_xtrack_interpolated['betx'] / (sigma_x * (sigma_x + sigma_y))  
        integrand_y = twiss_xtrack_interpolated['bety'] / (sigma_y * (sigma_x + sigma_y)) 
        
        dQx = - K_sc / (4 * np.pi) * np.trapz(integrand_x, x = twiss_xtrack_interpolated['s'])
        dQy = - K_sc / (4 * np.pi) * np.trapz(integrand_y, x = twiss_xtrack_interpolated['s'])
        
        return dQx, dQy



class IBS_Growth_Rates:
    """
    Class to calculate IBS growth rates for a given lattice for given lattice and beam parameters
    """
    @staticmethod
    def get_growth_rates(line, beamParams, also_calculate_kinetic=False, num_part=10_000):
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
            sigma_delta and bunch_length
        also_calculate_kinetic : bool
            whether to also calculate kinetic growth rates
        num_part : int
            number of macroparticles for xp.Part object for kinetic kick calculation
            
        Returns:
        --------
        growth_rates : dataclass
            containing Tx, Ty and Tz - growth rates in respective plane
        """

        beamparams = BeamParameters.from_line(line, n_part=beamParams.Nb)
        opticsparams = OpticsParameters.from_line(line)

        # Instantiate analytical Nagaitsev IBS class
        NIBS = NagaitsevIBS(beamparams, opticsparams)
        growth_rates = NIBS.growth_rates(beamParams.exn, 
                                         beamParams.eyn, 
                                         beamParams.sigma_delta, 
                                         beamParams.bunch_length,
                                         normalized_emittances=True)
        
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