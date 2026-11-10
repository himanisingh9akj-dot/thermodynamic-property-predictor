import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import (
    GASES,
    calculate_state,
    total_enthalpy_entropy,
    residual_properties,
)

EOS_MODELS = [
    "Peng-Robinson (PR)",
    "Soave-Redlich-Kwong (SRK)",
]


@pytest.mark.parametrize("eos", EOS_MODELS)
def test_methane_state(eos):
    state = calculate_state(
        300, 10, GASES["Methane"], eos,
        "Vapour-like (largest Z)"
    )

    assert state["Z"] > 0
    assert state["V"] > 0
    assert state["phi"] > 0
    assert state["f"] > 0


@pytest.mark.parametrize("eos", EOS_MODELS)
def test_fugacity_relation(eos):
    state = calculate_state(
        300, 10, GASES["Methane"], eos,
        "Vapour-like (largest Z)"
    )

    assert math.isclose(
        state["f"],
        state["phi"] * 10,
        rel_tol=1e-10
    )


@pytest.mark.parametrize("eos", EOS_MODELS)
def test_ideal_gas_limit(eos):
    state = calculate_state(
        300, 0.001, GASES["Methane"], eos,
        "Vapour-like (largest Z)"
    )

    assert abs(state["Z"] - 1) < 0.001
    assert abs(state["phi"] - 1) < 0.001


@pytest.mark.parametrize("eos", EOS_MODELS)
def test_enthalpy_entropy_calculation(eos):
    result = total_enthalpy_entropy(
        300, 10, GASES["Methane"], eos,
        "Vapour-like (largest Z)",
        298.15, 1
    )

    assert math.isfinite(result["H_total_change"])
    assert math.isfinite(result["S_total_change"])


@pytest.mark.parametrize("eos", EOS_MODELS)
def test_residual_properties(eos):
    Hres, Sres, state = residual_properties(
        300, 10, GASES["Methane"], eos,
        "Vapour-like (largest Z)"
    )

    assert math.isfinite(Hres)
    assert math.isfinite(Sres)


def test_invalid_temperature():
    with pytest.raises(ValueError):
        calculate_state(
            0, 10, GASES["Methane"],
            "Peng-Robinson (PR)",
            "Vapour-like (largest Z)"
        )


def test_invalid_pressure():
    with pytest.raises(ValueError):
        calculate_state(
            300, 0, GASES["Methane"],
            "Peng-Robinson (PR)",
            "Vapour-like (largest Z)"
        )


def test_multiple_roots_are_sorted():
    roots, params = calculate_roots_for_test()

    assert roots == sorted(roots)
    assert all(root > params["B"] for root in roots)


def calculate_roots_for_test():
    from app import z_roots

    return z_roots(
        150, 20, GASES["Methane"],
        "Peng-Robinson (PR)"
    )


def test_hs_reference_state():
    result = total_enthalpy_entropy(
        298.15, 1, GASES["Methane"],
        "Peng-Robinson (PR)",
        "Vapour-like (largest Z)",
        298.15, 1
    )

    assert abs(result["H_ideal_change"]) < 1e-10
    assert abs(result["S_ideal_change"]) < 1e-10
