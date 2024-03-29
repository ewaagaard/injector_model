# Injector Model for ions at CERN 

### Background

As of today, the LHC ion physics programme is mostly based on Pb ion collisions. Lighter ions species have the potential to yield higher nucleon-nucleon luminosities, as requested from the ALICE3 detector proposal. This proposal was recently discussed in the [2022 Letter of Intent](https://arxiv.org/abs/2211.02491). To achieve the requested luminosity increase, the CERN Ion Injector chain (consisting of Linac3, LEIR, PS and SPS) will need to provide significantly higher beam intensities with light ion beams as compared to the Pb ion beams. So far the operational experience with light ion beams inthe injectors is very limited and the beam dynamics limitations for these beams are not well known. 

In this repository, we develop a simulation tool for different ions based on the Mathematica notebook presented by [Bruce (2021)](https://indico.cern.ch/event/1085343/contributions/4563386/attachments/2326159/3964426/2021.10.12--LIU_ions--Run4_light_ion_scenarios.pdf). The tool is contained in a class to represent the Injector Chain, taking an ion species as input and returning the calculated bunch intensity into the LHC. We compare the output to estimates from the [Working Group 5 (WG5) report](https://cds.cern.ch/record/2650176). The Injector Model class calculates the propagated beam parameters and intensitities from LINAC3 through LEIR, PS and SPS into the LHC. The present Pb ion configuration is shown below. In this class, we explore different options with stripper foil locations and bunch splitting in the PS. 

![CERN_ion_injector_chain](https://github.com/ewaagaard/injector_model/assets/68541324/43abd382-aa74-4439-b864-bcf02f925fe5)


### Set-up

When using Python for scientific computing, it is important to be aware of dependencies and compatibility of different packages. This guide gives a good explanation: [Python dependency manager guide](https://aaltoscicomp.github.io/python-for-scicomp/dependencies/#dependency-management). An isolated environment allows installing packages without affecting the rest of your operating system or any other projects. A useful resource to handle virtual environments is [Anaconda](https://www.anaconda.com/) (or its lighter version Miniconda), when once installed has many useful commands of which many can be found in the [Conda cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf) 

To directly start calculating different ion performances with the `injector_model`, create an isolated virtual environment and perform a local install to use the `injector_model` freely. Once the repository is cloned and from inside the `injector_model` repository, run in the terminal:

```
conda create --name test_venv python=3.11 numpy pandas scipy matplotlib
conda activate test_venv
cd ..
python -m pip install -e injector_model
```
Then the different scripts in the folder `calculations` can be executed. The virtual environment can also be installed directly from the `requirements.txt`:

```
python -m pip install -r requirements.txt
```

In order to use the IBS module functions from M. Zampetakis on this [GitHub repository](https://github.com/MichZampetakis/IBS_for_Xsuite), which is a submodule in this repository, install the package in the Python virtual environment.
```
python -m pip install -e IBS_for_Xsuite/
```

### Usage 

- The Python class `CERN_Injector_Chain()` contained in `Injector_Chain.py` aims at modelling different ion species throughout the CERN accelerators. 
- The example script `Calculate_LHC_intensities.py` imports this class and generates both plots and table for different cases also discussed by Bruce (2021). 
- The directory `data` contains input data for different ion in `Ion_species.csv` and stable isotopes for all considered ion species, generated from `stable_isotopes.py`
- **Input**: pandas dataframe with ion data, can be loaded from `data/Ion_species.csv`. The intensity limit for a single ion can be calculated (specifying the ion type). The class is instantiated as in this example:
  ```python
  from injector_model import Injector_Chain
  
  injector_chain = CERN_Injector_Chain(ion_type, 
                        ion_data, 
                        nPulsesLEIR = 0, # 0 = automatic maximum number of pulses
                        LEIR_bunches = 2,
                        PS_splitting = 1,
                        account_for_SPS_transmission=True,
                        LEIR_PS_strip=True
                        )
  ```
- **Output**: after instantiation, the method `CERN_Injector_Chain.calculate_LHC_bunch_intensity_all_ion_species(save_csv=True, output_name='output1')` calculates the extracted bunch intensity from LEIR, PS and the SPS going into the LHC. This limit comes from either the total maximum intensity from the previous injector or from the space charge limit, for now calculated from the linear space charge tune shift corresponding to the maximum tune shift for present Pb beams. Also information about bunches, splitting, stripping and the relativistic $\gamma$ for each accelerator is calculated. If desired, this full csv table is saved in `Output`.   

### Example output plot for different scenarios

Various plots are generated in the example script `calculate_lhc_intensities.py`. The last one, comparing different cases, is presented here:

![3_LEIR_PS_stripping|200](https://github.com/ewaagaard/InjectorModel/assets/68541324/cacad841-63fd-4aff-8b58-e5ed89b971b5)
