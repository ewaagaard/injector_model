#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to investigate space charge (SC) tune shift evolution for different energies 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from injector_model import InjectorChain_full_SC

# Instantiate the injector chain object 
injectors = InjectorChain_full_SC('Pb')

#### PLOT THE DATA #######
SMALL_SIZE = 18
MEDIUM_SIZE = 22
BIGGER_SIZE = 28
plt.rcParams["font.family"] = "serif"
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)   # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
colors = ['green', 'blue', 'purple', 'brown', 'teal', 'coral', 'cyan', 'darkred']


def find_SC_tune_shift_for_energy(ion_type, Nbs=None, gamma_resolution=15):
    """
    For given standard beam parameters, find tune shift as a funtion of energy 
    Input: ion_type, Nbs: array of bunch intensity in LEIR, PS and SPS 
    """
    print("\nFinding SC tune shift for {}...\n".format(ion_type))
    
    # Initiate the correct ion type 
    injectors.init_ion(ion_type)
    injectors.simulate_injection()
    
    # If bunch intensities not given, take default values for Pb
    if Nbs is None:
        Nbs = np.array([injectors.Nb0_LEIR, injectors.Nb0_PS, injectors.Nb0_SPS])
    
    # Initiate energy range for all accelerators 
    LEIR_gammas = np.linspace(injectors.gamma_LEIR_inj, injectors.gamma_LEIR_extr, num = gamma_resolution)
    PS_gammas = np.linspace(injectors.gamma_PS_inj, injectors.gamma_PS_extr, num = gamma_resolution)
    SPS_gammas = np.linspace(injectors.gamma_SPS_inj, 50, num = gamma_resolution)
    
    Q_array = np.zeros([gamma_resolution, 6]) # empty array for all tunes
    
    # Iterate over the different energies 
    for i, gamma in enumerate(LEIR_gammas):
        print("\nEnergy nr {}\n".format(i))
        sigma_z_LEIR = 8.5 if ion_type == 'O' else injectors.sigma_z_LEIR  # special case for oxygen in LEIR 
        dQx_LEIR, dQy_LEIR = injectors.calculate_SC_tuneshift_for_LEIR(Nbs[0], LEIR_gammas[i], sigma_z = sigma_z_LEIR)
        dQx_PS, dQy_PS = injectors.calculate_SC_tuneshift_for_PS(Nbs[1], PS_gammas[i], sigma_z = injectors.sigma_z_PS)
        dQx_SPS, dQy_SPS = injectors.calculate_SC_tuneshift_for_SPS(Nbs[2], SPS_gammas[i], sigma_z = injectors.sigma_z_SPS)
        Q_array[i, :] = np.array([dQx_LEIR, dQy_LEIR, dQx_PS, dQy_PS, dQx_SPS, dQy_SPS])
        
    return Q_array, LEIR_gammas, PS_gammas, SPS_gammas


if __name__ == '__main__': 

    #### Pb ions ####
    Q_array_Pb, LEIR_gammas_Pb, PS_gammas_Pb, SPS_gammas_Pb = find_SC_tune_shift_for_energy('Pb')
    
    fig, (ax11, ax22, ax33) = plt.subplots(1, 3, figsize = (16,5))
    fig.suptitle("Pb", fontsize=29)
    ax11.plot(LEIR_gammas_Pb, Q_array_Pb[:, 0], linewidth= 3.5, ls='-', color='navy', label=r'LEIR $dQ_{x}$')
    ax11.plot(LEIR_gammas_Pb, Q_array_Pb[:, 1], linewidth= 3.5, ls=':', color='cyan', label=r'LEIR $dQ_{y}$')
    ax22.plot(PS_gammas_Pb, Q_array_Pb[:, 2], linewidth= 3.5, ls='-', color='maroon', label=r'PS $dQ_{x}$')
    ax22.plot(PS_gammas_Pb, Q_array_Pb[:, 3], linewidth= 3.5, ls=':', color='coral', label=r'PS $dQ_{y}$')
    ax33.plot(SPS_gammas_Pb, Q_array_Pb[:, 4], linewidth= 3.5, ls='-', color='forestgreen', label=r'SPS $dQ_{x}$')
    ax33.plot(SPS_gammas_Pb, Q_array_Pb[:, 5], linewidth= 3.5, ls=':', color='lime', label=r'SPS $dQ_{y}$')
    ax11.set_ylabel(r"Tune shift $dQ_{x,y}$")
    ax11.set_xlabel(r"Relativistic $\gamma$")
    ax22.set_xlabel(r"Relativistic $\gamma$")
    ax33.set_xlabel(r"Relativistic $\gamma$")
    ax33.set_xlim(0, 50)
    #ax.set_xscale('log')
    ax11.legend()
    ax22.legend()
    ax33.legend()
    #ax33.set_xscale('log')
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig.savefig('output/Pb_tune_shifts_over_gammas.png', dpi=250)

    # Create a dictionary to hold the data
    data_dict_Pb = {
        'LEIR_gamma': LEIR_gammas_Pb,
        'LEIR_dQ_x': Q_array_Pb[:, 0],
        'LEIR_dQ_y': Q_array_Pb[:, 1],
        'PS_gamma': PS_gammas_Pb,
        'PS_dQ_x': Q_array_Pb[:, 2],
        'PS_dQ_y': Q_array_Pb[:, 3],
        'SPS_gamma': SPS_gammas_Pb,
        'SPS_dQ_x': Q_array_Pb[:, 4],
        'SPS_dQ_y': Q_array_Pb[:, 5],
    }
    
    # Create the DataFrame from the dictionary
    df_Pb = pd.DataFrame(data_dict_Pb)
    df_Pb.to_csv('output/Pb_tune_shifts_over_gammas.csv')

    #### O ions ####
    Nbs_O = np.array([110e8, 88e8, 50e8])  # same intensities as from Bartosik and John 2021 report: (https://cds.cern.ch/record/2749453)
    
    Q_array_O, LEIR_gammas_O, PS_gammas_O, SPS_gammas_O = find_SC_tune_shift_for_energy('O', Nbs = Nbs_O)
    
    fig2, (ax11, ax22, ax33) = plt.subplots(1, 3, figsize = (16,5))
    fig2.suptitle("O", fontsize=29)
    ax11.plot(LEIR_gammas_O, Q_array_O[:, 0], linewidth= 3.5, ls='-', color='navy', label=r'LEIR $dQ_{x}$')
    ax11.plot(LEIR_gammas_O, Q_array_O[:, 1], linewidth= 3.5, ls=':', color='cyan', label=r'LEIR $dQ_{y}$')
    ax22.plot(PS_gammas_O, Q_array_O[:, 2], linewidth= 3.5, ls='-', color='maroon', label=r'PS $dQ_{x}$')
    ax22.plot(PS_gammas_O, Q_array_O[:, 3], linewidth= 3.5, ls=':', color='coral', label=r'PS $dQ_{y}$')
    ax33.plot(SPS_gammas_O, Q_array_O[:, 4], linewidth= 3.5, ls='-', color='forestgreen', label=r'SPS $dQ_{x}$')
    ax33.plot(SPS_gammas_O, Q_array_O[:, 5], linewidth= 3.5, ls=':', color='lime', label=r'SPS $dQ_{y}$')
    ax11.set_ylabel(r"Tune shift $dQ_{x,y}$")
    ax11.set_xlabel(r"Relativistic $\gamma$")
    ax22.set_xlabel(r"Relativistic $\gamma$")
    ax33.set_xlabel(r"Relativistic $\gamma$")
    ax33.set_xlim(0, 50)
    #ax.set_xscale('log')
    ax11.legend()
    ax22.legend()
    ax33.legend()
    #ax33.set_xscale('log')
    fig2.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig2.savefig('output/O_tune_shifts_over_gammas.png', dpi=250)
    
    # Create a dictionary to hold the data
    data_dict_O = {
        'LEIR_gamma': LEIR_gammas_O,
        'LEIR_dQ_x': Q_array_O[:, 0],
        'LEIR_dQ_y': Q_array_O[:, 1],
        'PS_gamma': PS_gammas_O,
        'PS_dQ_x': Q_array_O[:, 2],
        'PS_dQ_y': Q_array_O[:, 3],
        'SPS_gamma': SPS_gammas_O,
        'SPS_dQ_x': Q_array_O[:, 4],
        'SPS_dQ_y': Q_array_O[:, 5],
    }
    
    # Create the DataFrame from the dictionary
    df_O = pd.DataFrame(data_dict_O)
    df_O.to_csv('output/O_tune_shifts_over_gammas.csv')
