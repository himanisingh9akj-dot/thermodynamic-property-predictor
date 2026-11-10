# Pure Component Thermodynamic Property Predictor

## 1. Project Overview

The Pure Component Thermodynamic Property Predictor is an interactive
engineering application developed to estimate thermodynamic properties
of pure real gases using cubic equations of state.

The software implements the Peng-Robinson (PR) and Soave-Redlich-Kwong
(SRK) equations of state and provides calculations of PVT behaviour,
enthalpy, entropy, and fugacity.

## 2. Objectives

- Predict pressure-volume-temperature (PVT) behaviour of pure gases.
- Calculate enthalpy and entropy changes using ideal-gas and residual
  property contributions.
- Calculate fugacity coefficients and fugacity.
- Compare predictions from PR and SRK equations of state.
- Visualise thermodynamic behaviour through interactive plots.

## 3. Software Features

- Built-in database of ten common gases.
- Option to enter custom pure-component properties.
- Selection between PR and SRK equations of state.
- Vapour, liquid, and stable-root selection.
- PVT property calculations and graphical plots.
- Enthalpy and entropy calculations with reference-state inputs.
- Fugacity coefficient and fugacity calculations.
- Interactive interface developed using Streamlit.

## 4. Thermodynamic Models

The software uses two cubic equations of state:

1. Peng-Robinson (PR) equation of state.
2. Soave-Redlich-Kwong (SRK) equation of state.

The compressibility factor is obtained by solving the corresponding
cubic equation. Physically admissible real roots are considered for
phase selection.

The fugacity coefficient is calculated from the selected equation
of state, and fugacity is obtained from:

    f = phi * P

Enthalpy and entropy changes include ideal-gas contributions and
departure (residual) property contributions calculated using the
selected cubic equation of state.

## 5. Installation and Execution

Requirements:
- Python 3.10 or newer
- Packages listed in requirements.txt

Install dependencies:

    python3 -m pip install -r requirements.txt

Run the application:

    python3 -m streamlit run app.py

Open the local URL displayed in the terminal, usually:

    http://localhost:8501

## 6. Units

- Temperature: K
- Pressure: bar
- Molar volume: L/mol
- Enthalpy: J/mol
- Entropy: J/(mol K)
- Heat capacity: J/(mol K)
- Fugacity: bar
- Compressibility factor: dimensionless

## 7. Assumptions and Limitations

- The application is designed for pure-component calculations.
- Built-in ideal-gas heat capacities are approximate constant values
  near 300 K.
- Enthalpy and entropy are changes relative to a selected reference
  temperature and pressure, not absolute property values.
- The constant-Cp approximation may introduce errors over wide
  temperature ranges.
- Accuracy depends on the input critical properties and acentric
  factor.
- Phase selection and root interpretation require care near saturation
  and critical conditions.
- Predictions should be compared with trusted thermodynamic data
  before use in design or safety-critical calculations.

## 8. Testing

Run the automated test suite using:

    python3 -m pytest tests/test_core.py -v

The current test suite contains 14 passing tests covering both
equations of state, fugacity, ideal-gas limiting behaviour, enthalpy,
entropy, residual properties, input validation, root sorting, and
reference-state calculations.

See TEST_RESULTS.txt for the recorded test results.

## 9. Suggested Demonstration

1. Select methane and Peng-Robinson EOS at 300 K and 10 bar.
2. Demonstrate the PVT results and corresponding plots.
3. Open the H/S tab and show the enthalpy and entropy breakdown.
4. Open the Fugacity tab and demonstrate fugacity calculations.
5. Switch to SRK EOS to compare model predictions.
6. Demonstrate the custom-component input option.

## 10. Project Files

- app.py: Main Streamlit application and thermodynamic calculations.
- requirements.txt: Python package dependencies.
- tests/test_core.py: Automated test suite.
- TEST_RESULTS.txt: Test execution record.
- README.md: Project documentation.
