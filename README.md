# Injector Model for Ions at CERN

A Python package for simulating ion beam propagation through the CERN accelerator complex (Linac3 → LEIR → PS → SPS → LHC injection), with realistic physics modeling including space charge limitations, electron cooling, and operational scenarios.

## Overview

This numerical tool was developed to study future ion beams for the ALICE3 detector upgrade proposal, which among other experiments is interested in lighter ions for their physics program. It calculates bunch intensities through the CERN ion injector chain, accounting for:

- **Space charge limitations** in each accelerator stage
- **Electron cooling** effects in LEIR (may be critical for light ions)  
- **Bunch splitting in the PS**
- **Stripper foil placement** (LEIR-PS vs PS-SPS)
- **Rest gas transmission** and other realistic losses

The model is based on the framework by [Bruce (2021)](https://indico.cern.ch/event/1085343/contributions/4563386/attachments/2326159/3964426/2021.10.12--LIU_ions--Run4_light_ion_scenarios.pdf) and validated against [Working Group 5](https://cds.cern.ch/record/2650176) estimates.

![CERN_ion_injector_chain](https://github.com/ewaagaard/injector_model/assets/68541324/43abd382-aa74-4439-b864-bcf02f925fe5)

## Installation

Create a conda environment with specific package versions for compatibility:

```bash
conda create --name injector_model python=3.11 numpy==2.3.1 scipy==1.15.3
conda activate injector_model

# Install specific versions for optimal compatibility
pip install matplotlib==3.10.0 pandas==2.2.3
pip install xdeps==0.10.5 xfields==0.25.0 xobjects==0.5.0 xpart==0.23.0 xtrack==0.86.1

# Install the injector model package (from repository root)
python -m pip install -e .
```

## Quick Start

```python
from injector_model import InjectorChain

# Create injector chain instance
chain = InjectorChain(
    LEIR_bunches=2,                    # Number of bunches in LEIR
    PS_splitting=2,                    # PS splitting factor (1 or 2)
    LEIR_PS_strip=False,               # Stripping location (False = PS-SPS)
    account_for_LEIR_ecooling=True,    # Include electron cooling
    account_for_PS_rest_gas=True       # Include PS rest gas losses
)

# Calculate bunch intensities for all ion species
results = chain.calculate_LHC_bunch_intensity_all_ion_species()

# Or for a single ion (e.g., Pb)
pb_result = chain.calculate_LHC_bunch_intensity()
```

## Examples and Tutorials

Comprehensive Jupyter notebook examples are provided in the [`examples/`](examples/) directory:

- **[01_basic_usage.ipynb](examples/01_basic_usage.ipynb)**: Getting started, basic calculations, and output interpretation
- **[02_electron_cooling_effects.ipynb](examples/02_electron_cooling_effects.ipynb)**: Impact of LEIR electron cooling on different ion species
- **[03_bunch_splitting_scenarios.ipynb](examples/03_bunch_splitting_scenarios.ipynb)**: PS bunch splitting optimization strategies
- **[04_stripper_foil_comparison.ipynb](examples/04_stripper_foil_comparison.ipynb)**: LEIR-PS vs PS-SPS stripping analysis

See the [examples README](examples/README.md) for detailed descriptions and learning outcomes.

## Key Features

### Physics Modeling
- **Full space charge integrals** for LEIR, PS, and SPS using realistic lattice parameters
- **Ion-dependent electron cooling** with species-specific cooling times
- **Charge state evolution** through stripping scenarios
- **Transmission efficiencies** for each accelerator stage

### Analysis Capabilities  
- **Multi-ion species** calculations for systematic studies
- **Parameter optimization** for different physics goals
- **Comparative scenario analysis** with visualization tools
- **Performance prediction** for future ion programs

### Ion Species Coverage
Currently supports: **He, C, N, O, Ne, Mg, Ar, Ca, Kr, Xe, Pb** with extensible framework for additional species.

## Applications

- **Operational optimization**: Find optimal parameters for specific ion species
- **Performance prediction**: Estimate achievable bunch intensities  
- **Limitations analysis**: Understand space charge and other constraints
- **Upgrade impact assessment**: Evaluate potential improvements

## Key Results

From systematic analysis across ion species:

- **Electron cooling**: Factor 2-10 intensity improvements for light ions
- **PS splitting**: Optimal factors typically 1-2, species-dependent
- **Stripper placement**: Ion-specific optimization between LEIR-PS and PS-SPS
- **Space charge**: SPS limitations dominate for most high-intensity scenarios
- **Light ion potential**: Higher nucleon-nucleon rates achievable vs heavy ions

## Repository Structure

```
injector_model/
├── injector_model/           # Main package code
│   ├── injector_model.py    # Core InjectorChain class
│   ├── parameters_and_helpers.py  # Reference values and utilities  
│   └── space_charge_and_ibs.py    # Physics calculations
├── examples/                 # Jupyter notebook tutorials
├── data/                     # Ion species data and reference values
├── calculations/             # Analysis scripts and studies
└── tests/                    # Validation and test scripts
```

## Support and Development

- **Documentation**: See example notebooks and docstrings
- **Issues**: Report bugs through GitHub issues  
- **Contributing**: Follow existing analysis patterns, ensure reproducibility
- **Physics validation**: Results benchmarked against WG5 estimates and operational data

---

This package enables **quantitative, physics-based optimization** of ion beam production at CERN, supporting both current operations and future light ion physics programs.
