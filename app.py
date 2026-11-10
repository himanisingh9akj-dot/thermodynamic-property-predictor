
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ============================================================
# Pure-component thermodynamic property predictor
# Units: T [K], P [bar], V [L/mol], H [J/mol], S [J/mol/K]
# ============================================================

R_BAR = 0.08314462618       # L bar / (mol K)
R_J = 8.314462618           # J / (mol K)
LBAR_TO_J = 100.0           # 1 L bar = 100 J
SQRT2 = math.sqrt(2.0)

# Cp values below are approximate constant ideal-gas heat capacities
# near 300 K. They are editable in the Custom gas form.
GASES = {
    "Methane":  {"Tc": 190.564, "Pc": 45.99, "omega": 0.01142, "MW": 16.043, "Cp": 35.69},
    "Ethane":   {"Tc": 305.322, "Pc": 48.72, "omega": 0.0995,  "MW": 30.070, "Cp": 52.49},
    "Propane":  {"Tc": 369.830, "Pc": 42.48, "omega": 0.1523,  "MW": 44.097, "Cp": 73.60},
    "n-Butane": {"Tc": 425.125, "Pc": 37.96, "omega": 0.2002,  "MW": 58.124, "Cp": 98.5},
    "Nitrogen": {"Tc": 126.192, "Pc": 33.958, "omega": 0.0372,  "MW": 28.014, "Cp": 29.12},
    "Carbon dioxide": {"Tc": 304.128, "Pc": 73.773, "omega": 0.22394, "MW": 44.010, "Cp": 37.13},
    "Oxygen":   {"Tc": 154.581, "Pc": 50.430, "omega": 0.0222,  "MW": 31.999, "Cp": 29.38},
    "Hydrogen": {"Tc": 33.145,  "Pc": 12.964, "omega": -0.219,  "MW": 2.016,  "Cp": 28.84},
    "Water":    {"Tc": 647.096, "Pc": 220.64, "omega": 0.3443,  "MW": 18.015, "Cp": 33.58},
    "Ammonia":  {"Tc": 405.40,  "Pc": 113.53, "omega": 0.250,   "MW": 17.031, "Cp": 35.10},
}

def eos_parameters(T, P, gas, eos):
    Tc, Pc, w = gas["Tc"], gas["Pc"], gas["omega"]
    Tr = T / Tc
    if eos == "Peng-Robinson (PR)":
        m = 0.37464 + 1.54226*w - 0.26992*w*w
        ac = 0.45724 * R_BAR**2 * Tc**2 / Pc
        b = 0.07780 * R_BAR * Tc / Pc
        delta1, delta2 = 1.0 + SQRT2, 1.0 - SQRT2
    else:
        m = 0.480 + 1.574*w - 0.176*w*w
        ac = 0.42747 * R_BAR**2 * Tc**2 / Pc
        b = 0.08664 * R_BAR * Tc / Pc
        delta1, delta2 = 1.0, 0.0

    g = 1.0 + m*(1.0 - math.sqrt(Tr))
    alpha = g*g
    a = ac * alpha
    # Analytic derivative da/dT at constant critical properties
    dalpha_dT = -m*g / math.sqrt(T*Tc)
    da_dT = ac * dalpha_dT

    A = a*P/(R_BAR**2*T**2)
    B = b*P/(R_BAR*T)
    return {"a": a, "b": b, "da_dT": da_dT, "A": A, "B": B,
            "delta1": delta1, "delta2": delta2, "alpha": alpha, "m": m}

def z_roots(T, P, gas, eos):
    p = eos_parameters(T, P, gas, eos)
    A, B = p["A"], p["B"]
    if eos == "Peng-Robinson (PR)":
        coeff = [1.0, -(1.0-B), A-3.0*B*B-2.0*B, -(A*B-B*B-B**3)]
    else:
        coeff = [1.0, -1.0, A-B-B*B, -A*B]
    roots = np.roots(coeff)
    real_roots = sorted([float(x.real) for x in roots
                         if abs(x.imag) < 1e-7 and x.real > B + 1e-10])
    if not real_roots:
        raise ValueError("No physically admissible real compressibility-factor root was found. Check T, P and gas data.")
    return real_roots, p

def ln_fugacity_coefficient(Z, p, eos):
    A, B = p["A"], p["B"]
    if B <= 0 or Z <= B:
        raise ValueError("Invalid Z/B combination for fugacity calculation.")
    if eos == "Peng-Robinson (PR)":
        ratio = (Z + (1.0+SQRT2)*B) / (Z + (1.0-SQRT2)*B)
        return Z - 1.0 - math.log(Z-B) - A/(2.0*SQRT2*B)*math.log(ratio)
    else:
        return Z - 1.0 - math.log(Z-B) - (A/B)*math.log(1.0+B/Z)

def select_root(roots, p, eos, phase):
    if phase == "Vapour-like (largest Z)":
        return max(roots)
    if phase == "Liquid-like (smallest Z)":
        return min(roots)
    # At fixed T,P, compare residual Gibbs energy through ln(phi).
    return min(roots, key=lambda z: ln_fugacity_coefficient(z, p, eos))

def calculate_state(T, P, gas, eos, phase):
    if T <= 0 or P <= 0:
        raise ValueError("Temperature and pressure must both be greater than zero.")
    roots, p = z_roots(T, P, gas, eos)
    Z = select_root(roots, p, eos, phase)
    V = Z*R_BAR*T/P
    lnphi = ln_fugacity_coefficient(Z, p, eos)
    phi = math.exp(lnphi)
    fugacity = phi*P
    return {"T": T, "P": P, "Z": Z, "V": V, "lnphi": lnphi,
            "phi": phi, "f": fugacity, "roots": roots, "params": p}

def residual_properties(T, P, gas, eos, phase):
    state = calculate_state(T, P, gas, eos, phase)
    p, Z = state["params"], state["Z"]
    b, a, da = p["b"], p["a"], p["da_dT"]
    B = p["B"]
    d1, d2 = p["delta1"], p["delta2"]
    L = math.log((Z+d1*B)/(Z+d2*B))
    delta = d1-d2
    # General cubic-EOS departure expressions (J/mol and J/mol/K)
    Hres = LBAR_TO_J*(R_BAR*T*(Z-1.0) + (T*da-a)/(b*delta)*L)
    Sres = LBAR_TO_J*(R_BAR*math.log(Z-B) + da/(b*delta)*L)
    return Hres, Sres, state

def total_enthalpy_entropy(T, P, gas, eos, phase, Tref, Pref):
    if Tref <= 0 or Pref <= 0:
        raise ValueError("Reference temperature and pressure must be greater than zero.")
    Hres, Sres, state = residual_properties(T, P, gas, eos, phase)
    Cp = gas["Cp"]  # approximate constant ideal-gas Cp
    Hig = Cp*(T-Tref)
    Sig = Cp*math.log(T/Tref) - R_J*math.log(P/Pref)
    return {"H_ideal_change": Hig, "H_residual": Hres, "H_total_change": Hig+Hres,
            "S_ideal_change": Sig, "S_residual": Sres, "S_total_change": Sig+Sres,
            "state": state}

def build_custom_gas(name, Tc, Pc, omega, MW, Cp):
    vals = [Tc, Pc, MW, Cp]
    if any(v <= 0 for v in vals):
        raise ValueError("Tc, Pc, molecular weight and Cp must be positive.")
    return {"Tc": Tc, "Pc": Pc, "omega": omega, "MW": MW, "Cp": Cp}

# ------------------------- UI -------------------------
st.set_page_config(page_title="Pure Component Thermodynamic Predictor", page_icon="🧪", layout="wide")
st.title("🧪 Pure Component Thermodynamic Property Predictor")
st.caption("PVT behaviour • Enthalpy & Entropy • Pure-component Fugacity | Cubic Equations of State")

with st.sidebar:
    st.header("1. Component")
    gas_name = st.selectbox("Select pure gas", list(GASES.keys()) + ["Custom gas"])
    if gas_name == "Custom gas":
        custom_name = st.text_input("Gas name", "My gas")
        Tc = st.number_input("Critical temperature, Tc (K)", min_value=0.001, value=300.0, step=1.0)
        Pc = st.number_input("Critical pressure, Pc (bar)", min_value=0.001, value=50.0, step=1.0)
        omega = st.number_input("Acentric factor, ω", value=0.1, step=0.01, format="%.5f")
        MW = st.number_input("Molecular weight (g/mol)", min_value=0.001, value=30.0, step=1.0)
        Cp = st.number_input("Ideal-gas Cp near reference temperature (J/mol/K)", min_value=0.001, value=40.0, step=0.1)
        gas = build_custom_gas(custom_name, Tc, Pc, omega, MW, Cp)
        gas_label = custom_name
    else:
        gas = GASES[gas_name]
        gas_label = gas_name
    st.header("2. Model")
    eos = st.selectbox("Cubic equation of state", ["Peng-Robinson (PR)", "Soave-Redlich-Kwong (SRK)"])
    phase = st.selectbox("Root / phase selection", [
        "Vapour-like (largest Z)", "Liquid-like (smallest Z)", "Lowest fugacity (stable-root estimate)"
    ])
    st.caption("Use consistent units: K, bar, L/mol, J/mol and J/mol/K.")

st.subheader(f"Selected component: {gas_label}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Critical temperature", f"{gas['Tc']:.3f} K")
c2.metric("Critical pressure", f"{gas['Pc']:.3f} bar")
c3.metric("Acentric factor", f"{gas['omega']:.5f}")
c4.metric("Ideal-gas Cp (approx.)", f"{gas['Cp']:.2f} J/mol/K")

tab_pvt, tab_hs, tab_fug, tab_info = st.tabs([
    "① PVT Behaviour", "② Enthalpy & Entropy", "③ Pure Component Fugacity", "④ Model & Assumptions"
])

def show_state(state):
    a, b, c = st.columns(3)
    a.metric("Compressibility factor, Z", f"{state['Z']:.6f}")
    b.metric("Molar volume, V", f"{state['V']:.6f} L/mol")
    c.metric("Fugacity coefficient, φ", f"{state['phi']:.6f}")
    st.write("Admissible real roots:", ", ".join(f"{r:.6f}" for r in state["roots"]))

with tab_pvt:
    st.markdown("### Predict PVT behaviour from the selected cubic EOS")
    col1, col2 = st.columns(2)
    T_pvt = col1.number_input("Temperature (K)", min_value=1.0, value=300.0, step=5.0, key="pvtT")
    P_pvt = col2.number_input("Pressure (bar)", min_value=0.001, value=10.0, step=1.0, key="pvtP")
    if st.button("Calculate PVT state", key="pvtbutton"):
        try:
            state = calculate_state(T_pvt, P_pvt, gas, eos, phase)
            show_state(state)
            st.success("PVT calculation completed.")
        except Exception as e:
            st.error(str(e))
    st.markdown("#### P–V isotherm")
    pmin, pmax = st.columns(2)
    Pmin = pmin.number_input("Plot minimum pressure (bar)", min_value=0.001, value=1.0, key="plotpmin")
    Pmax = pmax.number_input("Plot maximum pressure (bar)", min_value=0.01, value=100.0, key="plotpmax")
    if st.button("Generate P–V curve", key="pvplot"):
        if Pmax <= Pmin:
            st.error("Maximum pressure must exceed minimum pressure.")
        else:
            ps = np.linspace(Pmin, Pmax, 100)
            vs, validp = [], []
            for pp in ps:
                try:
                    s = calculate_state(T_pvt, float(pp), gas, eos, phase)
                    vs.append(s["V"]); validp.append(pp)
                except Exception:
                    pass
            if len(vs) > 1:
                fig, ax = plt.subplots()
                ax.plot(vs, validp)
                ax.set_xlabel("Molar volume (L/mol)")
                ax.set_ylabel("Pressure (bar)")
                ax.set_title(f"{gas_label}: P–V isotherm at {T_pvt:g} K ({eos})")
                ax.grid(True)
                st.pyplot(fig)
            else:
                st.warning("Not enough valid points to draw this curve.")
    st.markdown("#### Z versus pressure")
    if st.button("Generate Z–P curve", key="zpplot"):
        ps = np.linspace(0.1, max(1.0, P_pvt*2.0), 100)
        zs, validp = [], []
        for pp in ps:
            try:
                zs.append(calculate_state(T_pvt, float(pp), gas, eos, phase)["Z"])
                validp.append(pp)
            except Exception:
                pass
        if len(zs) > 1:
            fig, ax = plt.subplots()
            ax.plot(validp, zs)
            ax.set_xlabel("Pressure (bar)")
            ax.set_ylabel("Compressibility factor, Z")
            ax.set_title(f"{gas_label}: Z–P at {T_pvt:g} K ({eos})")
            ax.grid(True)
            st.pyplot(fig)

with tab_hs:
    st.markdown("### Calculate enthalpy and entropy changes")
    st.info("H and S are reported as changes from the chosen reference state. The real-gas departure terms come from the selected cubic EOS. Ideal-gas contributions use an approximate constant Cp near 300 K.")
    col1, col2 = st.columns(2)
    T_hs = col1.number_input("State temperature (K)", min_value=1.0, value=300.0, step=5.0, key="hsT")
    P_hs = col2.number_input("State pressure (bar)", min_value=0.001, value=10.0, step=1.0, key="hsP")
    col3, col4 = st.columns(2)
    Tref = col3.number_input("Reference temperature (K)", min_value=1.0, value=298.15, step=1.0)
    Pref = col4.number_input("Reference pressure (bar)", min_value=0.001, value=1.0, step=0.1)
    if st.button("Calculate H and S", key="hsbutton"):
        try:
            result = total_enthalpy_entropy(T_hs, P_hs, gas, eos, phase, Tref, Pref)
            show_state(result["state"])
            st.markdown("#### Enthalpy")
            d1, d2, d3 = st.columns(3)
            d1.metric("Ideal-gas ΔH", f"{result['H_ideal_change']:.3f} J/mol")
            d2.metric("Residual H", f"{result['H_residual']:.3f} J/mol")
            d3.metric("Total ΔH", f"{result['H_total_change']:.3f} J/mol")
            st.markdown("#### Entropy")
            e1, e2, e3 = st.columns(3)
            e1.metric("Ideal-gas ΔS", f"{result['S_ideal_change']:.3f} J/mol/K")
            e2.metric("Residual S", f"{result['S_residual']:.3f} J/mol/K")
            e3.metric("Total ΔS", f"{result['S_total_change']:.3f} J/mol/K")
            st.success("H and S calculation completed.")
        except Exception as e:
            st.error(str(e))

with tab_fug:
    st.markdown("### Calculate pure-component fugacity")
    col1, col2 = st.columns(2)
    T_f = col1.number_input("Temperature (K)", min_value=1.0, value=300.0, step=5.0, key="fT")
    P_f = col2.number_input("Pressure (bar)", min_value=0.001, value=10.0, step=1.0, key="fP")
    if st.button("Calculate fugacity", key="fbutton"):
        try:
            state = calculate_state(T_f, P_f, gas, eos, phase)
            show_state(state)
            f1, f2 = st.columns(2)
            f1.metric("Fugacity coefficient, φ", f"{state['phi']:.8f}")
            f2.metric("Pure-component fugacity, f = φP", f"{state['f']:.8f} bar")
            st.write(f"ln(φ) = {state['lnphi']:.8f}")
            st.success("Pure-component fugacity calculation completed.")
        except Exception as e:
            st.error(str(e))

with tab_info:
    st.markdown("""
### Equations implemented
**Peng–Robinson (PR):**
\\[
P=\\frac{RT}{V-b}-\\frac{a(T)}{V(V+b)+b(V-b)}
\\]
\\[
Z^3-(1-B)Z^2+(A-3B^2-2B)Z-(AB-B^2-B^3)=0
\\]

**Soave–Redlich–Kwong (SRK):**
\\[
P=\\frac{RT}{V-b}-\\frac{a(T)}{V(V+b)}
\\]
\\[
Z^3-Z^2+(A-B-B^2)Z-AB=0
\\]

For both models, \\(A=aP/(R^2T^2)\\), \\(B=bP/(RT)\\), and \\(V=ZRT/P\\).

### Property calculations
\\[
H^R=RT(Z-1)+\\frac{T(da/dT)-a}{b(\\delta_1-\\delta_2)}
\\ln\\left(\\frac{Z+\\delta_1B}{Z+\\delta_2B}\\right)
\\]
\\[
S^R=R\\ln(Z-B)+\\frac{da/dT}{b(\\delta_1-\\delta_2)}
\\ln\\left(\\frac{Z+\\delta_1B}{Z+\\delta_2B}\\right)
\\]
The implementation converts L·bar to joules. Ideal-gas contributions use constant Cp:
\\[
\\Delta H^{ig}=C_p(T-T_{ref})
\\]
\\[
\\Delta S^{ig}=C_p\\ln(T/T_{ref})-R\\ln(P/P_{ref})
\\]
\\[
f=\\phi P
\\]

### Important assumptions and limitations
- This is a **pure-component**, single-phase cubic-EOS calculator. It is not a mixture or VLE simulator.
- The built-in ideal-gas Cp values are approximate constants near 300 K, not high-accuracy temperature-dependent correlations. H and S are therefore engineering estimates. For improved accuracy, enter validated Cp data / implement temperature-dependent Cp correlations.
- The "lowest fugacity" root is a stable-root estimate. Near saturation, a rigorous phase-equilibrium calculation requires checking saturation fugacity equality.
- The built-in gas list is finite. The Custom gas option supports another pure gas when its Tc, Pc, acentric factor, molecular weight and Cp are known.
- Results depend on the accuracy of critical constants, acentric factor and Cp inputs.
""")

st.divider()
st.caption("Educational engineering software. Verify property data, units, phase selection and results against trusted references before design use.")
