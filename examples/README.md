# Injector Model Examples

This folder contains Jupyter notebook examples demonstrating the functionality and applications of the `injector_model` package for CERN ion accelerator physics.

## Overview

The `injector_model` package simulates **ion beam intensity propagation** through the CERN injector complex (Linac3 → LEIR → PS → SPS → LHC), accounting for realistic physics limitations including space charge effects, electron cooling, rest gas transmission, and various operational scenarios.

These examples are designed to:
- Demonstrate core functionality of the package
- Provide realistic physics examples for accelerator operations
- Show best practices for analysis and visualization
- Guide optimization studies for different ion species

## Notebook Contents

### [01_basic_usage.ipynb](01_basic_usage.ipynb)
**Getting Started with the Injector Model**

- Basic setup and configuration
- Single ion calculations (Pb example)
- Multi-ion species analysis
- Understanding output parameters
- Basic visualization techniques
- Interpreting space charge limitations

**Learning Outcomes:**
- How to create and configure an `InjectorChain` instance
- Understanding the main output parameters
- Basic plotting and data visualization
- Identifying limiting factors in each accelerator stage

---

### [02_electron_cooling_effects.ipynb](02_electron_cooling_effects.ipynb)
**Impact of LEIR Electron Cooling**

- Comparison of scenarios with/without electron cooling
- Ion-species-dependent cooling benefits
- LEIR injection capability analysis
- Physics insights into cooling mechanisms
- Performance improvement quantification

**Learning Outcomes:**
- Critical importance of electron cooling for light ions
- How cooling enables multiple LEIR injections
- Ion-dependent optimization strategies
- Quantifying intensity improvements (factor 2-10 possible)

---

### [03_bunch_splitting_scenarios.ipynb](03_bunch_splitting_scenarios.ipynb)
**PS Bunch Splitting Optimization**

- Systematic comparison of splitting factors (1 vs 2)
- Trade-offs between bunch intensity and total current
- Space charge relief through splitting
- Ion-specific optimization analysis
- Operational strategy recommendations

**Learning Outcomes:**
- Understanding the bunch intensity vs. total current trade-off
- How splitting provides space charge relief
- Finding optimal splitting for different physics goals
- Species-dependent splitting strategies

---

### [04_stripper_foil_comparison.ipynb](04_stripper_foil_comparison.ipynb)
**Stripper Foil Placement Analysis**

- PS-SPS vs. LEIR-PS stripping comparison
- Charge state evolution analysis
- Space charge implications in different accelerators
- Ion-specific performance comparison
- Operational recommendations

**Learning Outcomes:**
- Impact of stripper foil placement on beam parameters
- How charge state affects space charge in each accelerator stage
- Ion-dependent optimization requirements for stripping strategies

## Quick Start Guide

### Prerequisites

Follow the installation instructions in the main [README.md](../README.md) for the complete setup with specific package versions:

```bash
conda create --name injector_model python=3.11 numpy==2.3.1 scipy==1.15.3
conda activate injector_model
pip install matplotlib==3.10.0 pandas==2.2.3
pip install xdeps==0.10.5 xfields==0.25.0 xobjects==0.5.0 xpart==0.23.0 xtrack==0.86.1
pip install jupyter  # For running the notebooks
```

### Running the Examples

```bash
# Navigate to the examples directory
cd examples

# Start Jupyter notebook
jupyter notebook

# Or use Jupyter Lab (if installed: pip install jupyterlab)
jupyter lab
```

Then open any of the `.ipynb` files in your browser.

## Notebook Overview

The examples progress from basic usage to advanced analysis techniques:

1. **Basic Usage**: Start here to understand the core functionality and model flow
2. **Electron Cooling**: Learn about critical physics effects for light ions  
3. **Bunch Splitting**: Optimize PS splitting strategies
4. **Stripper Foil**: Compare different stripping configurations

Each notebook builds on previous concepts while being self-contained for reference.

## Physics Concepts and Analysis Techniques

### Core Concepts:
- Space charge limitations in circular accelerators
- Electron cooling physics and applications
- Ion stripping and charge state optimization
- Beam dynamics in multi-stage accelerator systems
- Using the `InjectorChain` class effectively


## Key Results and Physics Insights

### Main Findings from the Analysis:

1. **Electron Cooling Impact**: Critical for light ions, enabling factor 2-10 intensity improvements depending on species

2. **PS Splitting Analysis**: Optimal splitting factors typically 1-2 for most ions, with trade-offs between bunch intensity and total current

3. **Stripper Foil Optimization**: Ion-dependent choice between LEIR-PS and PS-SPS configurations based on charge state evolution

4. **Space Charge Limitations**: SPS space charge represents the primary bottleneck for high-intensity operation across most ion species

5. **Ion Species Performance**: Lighter ions generally achieve higher nucleon-nucleon collision rates, with species-specific optimization requirements

### Practical Recommendations:
- **Baseline Configuration**: LEIR electron cooling + PS splitting optimization + strategic stripper foil placement
- **Light Ion Operation**: Electron cooling mandatory, higher splitting beneficial for space charge relief
- **Heavy Ion Operation**: Lower splitting preferred, focus on operational stability and transmission

## Advanced Usage

### Customizing Parameters:
```python
# Example: Custom configuration for specific studies
injector_chain = InjectorChain(
    LEIR_bunches=2,                    # Number of bunches in LEIR
    PS_splitting=2,                    # PS splitting factor (1 or 2)
    LEIR_PS_strip=False,               # Stripping location
    account_for_LEIR_ecooling=True,    # Include electron cooling
    account_for_PS_rest_gas=True       # Include PS transmission
)
```

### Batch Analysis:
```python
# Analyze multiple configurations efficiently
configurations = [
    {'PS_splitting': 1, 'LEIR_PS_strip': False},
    {'PS_splitting': 2, 'LEIR_PS_strip': False},
    {'PS_splitting': 1, 'LEIR_PS_strip': True},
]

results = {}
for config in configurations:
    chain = InjectorChain(**config)
    results[str(config)] = chain.calculate_LHC_bunch_intensity_all_ion_species()
```

## Further Reading

- **Package Documentation**: See main [README.md](../README.md) for package overview and installation
- **Ion Data Sources**: See `data/` directory for input parameters and reference values
- **ALICE3 Proposal**: [Letter of Intent](https://arxiv.org/abs/2211.02491) for future light ion physics program context

## Contributing

Follow the existing analysis style with clear physics explanations, test thoroughly, and document the physics background and interpretation.

---

These examples provide a foundation for understanding and optimizing ion beam performance in the CERN accelerator complex, enabling quantitative, physics-based decision making for operations, experiments, and future physics programs.
