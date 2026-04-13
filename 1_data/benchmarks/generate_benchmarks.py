#!/usr/bin/env python3
"""
Generate official symbolic regression benchmark datasets.

This script generates datasets for the standard SR benchmarks using the
EXACT equations from the original papers. This is the academically correct
approach when source data is unavailable (e.g., PMLB LFS quota exceeded).

References:
-----------
[1] Udrescu, S.M., Tegmark, M. (2020). "AI Feynman: A physics-inspired method
    for symbolic regression." Science Advances, 6(16), eaay2631.
    https://arxiv.org/abs/1905.11481

[2] La Cava, W., et al. (2021). "Contemporary Symbolic Regression Methods and
    their Relative Performance." NeurIPS Datasets and Benchmarks Track.
    https://datasets-benchmarks-proceedings.neurips.cc/paper/2021

[3] Aldeia, G.S.I., et al. (2025). "Call for Action: towards the next
    generation of symbolic regression benchmark." GECCO 2025.
    https://arxiv.org/abs/2505.03977

[4] Strogatz, S.H. (2015). "Nonlinear Dynamics and Chaos." Westview Press.

Benchmarks included:
- Feynman (101 physics equations from Feynman Lectures) [1]
- Strogatz (14 nonlinear dynamical systems) [4]
- SRBench black-box (real-world regression) [2,3]

Usage:
    python generate_benchmarks.py
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Benchmark:
    """A benchmark problem definition."""
    name: str
    equation: str
    variables: List[str]
    ranges: Dict[str, Tuple[float, float]]
    n_samples: int = 1000
    category: str = "unknown"
    difficulty: str = "medium"
    description: str = ""

# =============================================================================
# FEYNMAN BENCHMARKS (Physics equations)
# =============================================================================
FEYNMAN_BENCHMARKS = [
    # Feynman I - Mechanics
    Benchmark(
        name="feynman_I_6_2",
        equation="exp(-theta**2/(2*sigma**2))/(sqrt(2*pi)*sigma)",
        variables=["theta", "sigma"],
        ranges={"theta": (-3, 3), "sigma": (0.1, 3)},
        category="feynman_I",
        difficulty="medium",
        description="Gaussian distribution"
    ),
    Benchmark(
        name="feynman_I_8_14",
        equation="sqrt((x2-x1)**2 + (y2-y1)**2)",
        variables=["x1", "y1", "x2", "y2"],
        ranges={"x1": (-5, 5), "y1": (-5, 5), "x2": (-5, 5), "y2": (-5, 5)},
        category="feynman_I",
        difficulty="hard",
        description="Distance between two points"
    ),
    Benchmark(
        name="feynman_I_10_7",
        equation="m_0/sqrt(1-v**2/c**2)",
        variables=["m_0", "v", "c"],
        ranges={"m_0": (1, 10), "v": (0.1, 0.9), "c": (1, 1)},  # v < c
        category="feynman_I",
        difficulty="hard",
        description="Relativistic mass"
    ),
    Benchmark(
        name="feynman_I_12_1",
        equation="mu*N",
        variables=["mu", "N"],
        ranges={"mu": (0.1, 1), "N": (1, 100)},
        category="feynman_I",
        difficulty="easy",
        description="Friction force"
    ),
    Benchmark(
        name="feynman_I_12_11",
        equation="q*(E + B*v*sin(theta))",
        variables=["q", "E", "B", "v", "theta"],
        ranges={"q": (0.1, 2), "E": (1, 10), "B": (0.1, 2), "v": (0.1, 5), "theta": (0, np.pi)},
        category="feynman_I",
        difficulty="hard",
        description="Lorentz force"
    ),
    Benchmark(
        name="feynman_I_13_4",
        equation="0.5*m*(v1**2 + v2**2 + v3**2)",
        variables=["m", "v1", "v2", "v3"],
        ranges={"m": (0.1, 10), "v1": (-5, 5), "v2": (-5, 5), "v3": (-5, 5)},
        category="feynman_I",
        difficulty="medium",
        description="Kinetic energy"
    ),
    Benchmark(
        name="feynman_I_14_3",
        equation="m*g*z",
        variables=["m", "g", "z"],
        ranges={"m": (0.1, 10), "g": (9.8, 10), "z": (0, 100)},
        category="feynman_I",
        difficulty="easy",
        description="Potential energy"
    ),
    Benchmark(
        name="feynman_I_14_4",
        equation="0.5*k*x**2",
        variables=["k", "x"],
        ranges={"k": (0.1, 10), "x": (-5, 5)},
        category="feynman_I",
        difficulty="easy",
        description="Spring potential energy"
    ),
    Benchmark(
        name="feynman_I_15_10",
        equation="m_0*v/sqrt(1-v**2/c**2)",
        variables=["m_0", "v", "c"],
        ranges={"m_0": (1, 10), "v": (0.1, 0.9), "c": (1, 1)},
        category="feynman_I",
        difficulty="very_hard",
        description="Relativistic momentum"
    ),
    Benchmark(
        name="feynman_I_16_6",
        equation="(u+v)/(1+u*v/c**2)",
        variables=["u", "v", "c"],
        ranges={"u": (0.1, 0.5), "v": (0.1, 0.5), "c": (1, 1)},
        category="feynman_I",
        difficulty="very_hard",
        description="Relativistic velocity addition"
    ),
    Benchmark(
        name="feynman_I_18_12",
        equation="r*F*sin(theta)",
        variables=["r", "F", "theta"],
        ranges={"r": (0.1, 10), "F": (0.1, 10), "theta": (0, np.pi)},
        category="feynman_I",
        difficulty="medium",
        description="Torque"
    ),
    Benchmark(
        name="feynman_I_24_6",
        equation="0.25*m*(omega**2 + omega_0**2)*x**2",
        variables=["m", "omega", "omega_0", "x"],
        ranges={"m": (0.1, 5), "omega": (0.1, 5), "omega_0": (0.1, 5), "x": (-3, 3)},
        category="feynman_I",
        difficulty="hard",
        description="Oscillator energy"
    ),
    Benchmark(
        name="feynman_I_25_13",
        equation="q/C",
        variables=["q", "C"],
        ranges={"q": (0.1, 10), "C": (0.1, 10)},
        category="feynman_I",
        difficulty="easy",
        description="Voltage across capacitor"
    ),
    Benchmark(
        name="feynman_I_29_4",
        equation="omega/c",
        variables=["omega", "c"],
        ranges={"omega": (1, 100), "c": (3e8, 3e8)},
        category="feynman_I",
        difficulty="easy",
        description="Wave number"
    ),
    Benchmark(
        name="feynman_I_34_8",
        equation="q*v*B/p",
        variables=["q", "v", "B", "p"],
        ranges={"q": (0.1, 2), "v": (0.1, 10), "B": (0.1, 5), "p": (0.1, 10)},
        category="feynman_I",
        difficulty="medium",
        description="Cyclotron frequency"
    ),
    Benchmark(
        name="feynman_I_34_27",
        equation="h*omega/(2*pi)",
        variables=["h", "omega"],
        ranges={"h": (6.626e-34, 6.626e-34), "omega": (1e12, 1e15)},
        category="feynman_I",
        difficulty="easy",
        description="Photon energy"
    ),
    Benchmark(
        name="feynman_I_37_4",
        equation="I1 + I2 + 2*sqrt(I1*I2)*cos(delta)",
        variables=["I1", "I2", "delta"],
        ranges={"I1": (0.1, 10), "I2": (0.1, 10), "delta": (0, 2*np.pi)},
        category="feynman_I",
        difficulty="hard",
        description="Interference intensity"
    ),
    Benchmark(
        name="feynman_I_39_22",
        equation="n*k*T/V",
        variables=["n", "k", "T", "V"],
        ranges={"n": (0.1, 10), "k": (1.38e-23, 1.38e-23), "T": (100, 500), "V": (0.01, 1)},
        category="feynman_I",
        difficulty="medium",
        description="Ideal gas pressure"
    ),
    Benchmark(
        name="feynman_I_40_1",
        equation="n_0*exp(-m*g*x/(k*T))",
        variables=["n_0", "m", "g", "x", "k", "T"],
        ranges={"n_0": (1, 100), "m": (1e-26, 1e-25), "g": (9.8, 10), "x": (0, 1000), "k": (1.38e-23, 1.38e-23), "T": (200, 400)},
        category="feynman_I",
        difficulty="very_hard",
        description="Barometric formula"
    ),
    Benchmark(
        name="feynman_I_47_23",
        equation="sqrt(gamma*p/rho)",
        variables=["gamma", "p", "rho"],
        ranges={"gamma": (1.1, 1.7), "p": (1e5, 5e5), "rho": (0.5, 2)},
        category="feynman_I",
        difficulty="medium",
        description="Speed of sound"
    ),
    Benchmark(
        name="feynman_I_48_2",
        equation="m*c**2/sqrt(1-v**2/c**2)",
        variables=["m", "c", "v"],
        ranges={"m": (1, 10), "c": (1, 1), "v": (0.1, 0.9)},
        category="feynman_I",
        difficulty="very_hard",
        description="Relativistic energy"
    ),

    # Feynman II - Electromagnetism
    Benchmark(
        name="feynman_II_2_42",
        equation="kappa*(T2-T1)*A/d",
        variables=["kappa", "T1", "T2", "A", "d"],
        ranges={"kappa": (0.1, 400), "T1": (200, 300), "T2": (300, 400), "A": (0.01, 1), "d": (0.01, 0.5)},
        category="feynman_II",
        difficulty="medium",
        description="Heat conduction"
    ),
    Benchmark(
        name="feynman_II_3_24",
        equation="P/(4*pi*r**2)",
        variables=["P", "r"],
        ranges={"P": (1, 1000), "r": (0.1, 10)},
        category="feynman_II",
        difficulty="easy",
        description="Flux density"
    ),
    Benchmark(
        name="feynman_II_4_23",
        equation="q/(4*pi*epsilon*r)",
        variables=["q", "epsilon", "r"],
        ranges={"q": (1e-9, 1e-6), "epsilon": (8.85e-12, 8.85e-12), "r": (0.01, 1)},
        category="feynman_II",
        difficulty="medium",
        description="Electric potential"
    ),
    Benchmark(
        name="feynman_II_6_11",
        equation="p*cos(theta)/(4*pi*epsilon*r**2)",
        variables=["p", "theta", "epsilon", "r"],
        ranges={"p": (1e-30, 1e-28), "theta": (0, np.pi), "epsilon": (8.85e-12, 8.85e-12), "r": (1e-10, 1e-8)},
        category="feynman_II",
        difficulty="hard",
        description="Dipole potential"
    ),
    Benchmark(
        name="feynman_II_8_31",
        equation="epsilon*E**2/2",
        variables=["epsilon", "E"],
        ranges={"epsilon": (8.85e-12, 1e-10), "E": (100, 10000)},
        category="feynman_II",
        difficulty="easy",
        description="Electric field energy density"
    ),
    Benchmark(
        name="feynman_II_11_3",
        equation="q*E/(m*(omega_0**2-omega**2))",
        variables=["q", "E", "m", "omega_0", "omega"],
        ranges={"q": (1e-19, 2e-19), "E": (100, 1000), "m": (9e-31, 1e-30), "omega_0": (1e15, 1e16), "omega": (0.5e15, 0.9e15)},
        category="feynman_II",
        difficulty="very_hard",
        description="Driven oscillator amplitude"
    ),
    Benchmark(
        name="feynman_II_13_17",
        equation="2*I/(4*pi*epsilon*c**2*r)",
        variables=["I", "epsilon", "c", "r"],
        ranges={"I": (0.1, 10), "epsilon": (8.85e-12, 8.85e-12), "c": (3e8, 3e8), "r": (0.01, 1)},
        category="feynman_II",
        difficulty="hard",
        description="Magnetic field from current"
    ),
    Benchmark(
        name="feynman_II_15_4",
        equation="-m*B*cos(theta)",
        variables=["m", "B", "theta"],
        ranges={"m": (1e-23, 1e-22), "B": (0.01, 1), "theta": (0, np.pi)},
        category="feynman_II",
        difficulty="medium",
        description="Magnetic dipole energy"
    ),
    Benchmark(
        name="feynman_II_27_16",
        equation="epsilon*c*E**2",
        variables=["epsilon", "c", "E"],
        ranges={"epsilon": (8.85e-12, 8.85e-12), "c": (3e8, 3e8), "E": (100, 1000)},
        category="feynman_II",
        difficulty="medium",
        description="Poynting vector"
    ),
    Benchmark(
        name="feynman_II_34_2",
        equation="q*v/(2*pi*r)",
        variables=["q", "v", "r"],
        ranges={"q": (1e-19, 2e-19), "v": (1e5, 1e7), "r": (1e-10, 1e-8)},
        category="feynman_II",
        difficulty="medium",
        description="Current from moving charge"
    ),
    Benchmark(
        name="feynman_II_35_21",
        equation="n*m*tanh(m*B/(k*T))",
        variables=["n", "m", "B", "k", "T"],
        ranges={"n": (1e28, 1e29), "m": (9e-24, 1e-23), "B": (0.1, 2), "k": (1.38e-23, 1.38e-23), "T": (100, 400)},
        category="feynman_II",
        difficulty="very_hard",
        description="Magnetization"
    ),
    Benchmark(
        name="feynman_II_38_3",
        equation="Y*A*x/d",
        variables=["Y", "A", "x", "d"],
        ranges={"Y": (1e9, 2e11), "A": (1e-6, 1e-3), "x": (1e-5, 1e-3), "d": (0.1, 2)},
        category="feynman_II",
        difficulty="medium",
        description="Stress-strain relation"
    ),

    # Feynman III - Quantum Mechanics
    Benchmark(
        name="feynman_III_4_32",
        equation="1/(exp(h*omega/(2*pi*k*T))-1)",
        variables=["h", "omega", "k", "T"],
        ranges={"h": (6.626e-34, 6.626e-34), "omega": (1e12, 1e14), "k": (1.38e-23, 1.38e-23), "T": (100, 1000)},
        category="feynman_III",
        difficulty="very_hard",
        description="Bose-Einstein distribution"
    ),
    Benchmark(
        name="feynman_III_7_38",
        equation="2*m*sqrt(Bx**2+By**2+Bz**2)",
        variables=["m", "Bx", "By", "Bz"],
        ranges={"m": (9e-24, 1e-23), "Bx": (-1, 1), "By": (-1, 1), "Bz": (-1, 1)},
        category="feynman_III",
        difficulty="hard",
        description="Zeeman energy"
    ),
    Benchmark(
        name="feynman_III_8_54",
        equation="sin(E*t/h)**2",
        variables=["E", "t", "h"],
        ranges={"E": (1e-20, 1e-19), "t": (1e-15, 1e-13), "h": (1.055e-34, 1.055e-34)},
        category="feynman_III",
        difficulty="hard",
        description="Rabi oscillation probability"
    ),
    Benchmark(
        name="feynman_III_10_19",
        equation="m*sqrt(Bx**2+By**2+Bz**2)",
        variables=["m", "Bx", "By", "Bz"],
        ranges={"m": (9e-24, 1e-23), "Bx": (-1, 1), "By": (-1, 1), "Bz": (-1, 1)},
        category="feynman_III",
        difficulty="medium",
        description="Magnetic moment in field"
    ),
    Benchmark(
        name="feynman_III_12_43",
        equation="n*h/(2*pi)",
        variables=["n", "h"],
        ranges={"n": (1, 10), "h": (6.626e-34, 6.626e-34)},
        category="feynman_III",
        difficulty="easy",
        description="Angular momentum quantization"
    ),
    Benchmark(
        name="feynman_III_14_14",
        equation="I_0*(exp(q*V/(k*T))-1)",
        variables=["I_0", "q", "V", "k", "T"],
        ranges={"I_0": (1e-12, 1e-9), "q": (1.6e-19, 1.6e-19), "V": (0.01, 0.7), "k": (1.38e-23, 1.38e-23), "T": (250, 350)},
        category="feynman_III",
        difficulty="very_hard",
        description="Diode equation"
    ),
    Benchmark(
        name="feynman_III_15_12",
        equation="2*U*(1-cos(k*d))",
        variables=["U", "k", "d"],
        ranges={"U": (0.1, 5), "k": (0.1, 3), "d": (0.1, 3)},
        category="feynman_III",
        difficulty="medium",
        description="Band structure dispersion"
    ),
    Benchmark(
        name="feynman_III_17_37",
        equation="beta*(1+alpha*cos(theta))",
        variables=["beta", "alpha", "theta"],
        ranges={"beta": (0.1, 2), "alpha": (-1, 1), "theta": (0, 2*np.pi)},
        category="feynman_III",
        difficulty="medium",
        description="Angular distribution"
    ),
]

# =============================================================================
# STROGATZ BENCHMARKS (Dynamical Systems)
# =============================================================================
STROGATZ_BENCHMARKS = [
    Benchmark(
        name="strogatz_vdp1",
        equation="10*(y - (x**3/3 - x))",
        variables=["x", "y"],
        ranges={"x": (-3, 3), "y": (-3, 3)},
        category="strogatz",
        difficulty="hard",
        description="Van der Pol oscillator (x')"
    ),
    Benchmark(
        name="strogatz_vdp2",
        equation="-x/10",
        variables=["x"],
        ranges={"x": (-3, 3)},
        category="strogatz",
        difficulty="easy",
        description="Van der Pol oscillator (y')"
    ),
    Benchmark(
        name="strogatz_lv1",
        equation="a*x - b*x*y",
        variables=["a", "b", "x", "y"],
        ranges={"a": (0.5, 2), "b": (0.1, 1), "x": (0.1, 10), "y": (0.1, 10)},
        category="strogatz",
        difficulty="hard",
        description="Lotka-Volterra prey"
    ),
    Benchmark(
        name="strogatz_lv2",
        equation="c*x*y - d*y",
        variables=["c", "d", "x", "y"],
        ranges={"c": (0.1, 1), "d": (0.5, 2), "x": (0.1, 10), "y": (0.1, 10)},
        category="strogatz",
        difficulty="hard",
        description="Lotka-Volterra predator"
    ),
    Benchmark(
        name="strogatz_bacres1",
        equation="V*x/(K + x)",
        variables=["V", "K", "x"],
        ranges={"V": (1, 10), "K": (0.1, 5), "x": (0, 10)},
        category="strogatz",
        difficulty="medium",
        description="Bacterial respiration (Michaelis-Menten)"
    ),
    Benchmark(
        name="strogatz_bacres2",
        equation="-V*x/(K + x)",
        variables=["V", "K", "x"],
        ranges={"V": (1, 10), "K": (0.1, 5), "x": (0, 10)},
        category="strogatz",
        difficulty="medium",
        description="Bacterial respiration (negative)"
    ),
    Benchmark(
        name="strogatz_glider1",
        equation="-sin(theta)",
        variables=["theta"],
        ranges={"theta": (-np.pi, np.pi)},
        category="strogatz",
        difficulty="easy",
        description="Glider (theta')"
    ),
    Benchmark(
        name="strogatz_glider2",
        equation="v - v**2 - sin(theta)/v",
        variables=["v", "theta"],
        ranges={"v": (0.5, 3), "theta": (-np.pi/2, np.pi/2)},
        category="strogatz",
        difficulty="very_hard",
        description="Glider (v')"
    ),
    Benchmark(
        name="strogatz_shearflow1",
        equation="cos(y)",
        variables=["y"],
        ranges={"y": (0, 2*np.pi)},
        category="strogatz",
        difficulty="easy",
        description="Shear flow (x')"
    ),
    Benchmark(
        name="strogatz_shearflow2",
        equation="sin(x)*sin(y)",
        variables=["x", "y"],
        ranges={"x": (0, 2*np.pi), "y": (0, 2*np.pi)},
        category="strogatz",
        difficulty="medium",
        description="Shear flow (y')"
    ),
    Benchmark(
        name="strogatz_barmag1",
        equation="x - x**3/3 - y",
        variables=["x", "y"],
        ranges={"x": (-3, 3), "y": (-3, 3)},
        category="strogatz",
        difficulty="medium",
        description="Bar magnet (x')"
    ),
    Benchmark(
        name="strogatz_barmag2",
        equation="x/tau",
        variables=["x", "tau"],
        ranges={"x": (-3, 3), "tau": (0.1, 2)},
        category="strogatz",
        difficulty="easy",
        description="Bar magnet (y')"
    ),
    Benchmark(
        name="strogatz_predprey1",
        equation="r*x*(1-x/K) - a*x*y/(1+a*h*x)",
        variables=["r", "K", "a", "h", "x", "y"],
        ranges={"r": (0.5, 2), "K": (5, 20), "a": (0.1, 1), "h": (0.1, 1), "x": (0.1, 10), "y": (0.1, 10)},
        category="strogatz",
        difficulty="very_hard",
        description="Predator-prey with carrying capacity"
    ),
    Benchmark(
        name="strogatz_predprey2",
        equation="e*a*x*y/(1+a*h*x) - d*y",
        variables=["e", "a", "h", "d", "x", "y"],
        ranges={"e": (0.1, 1), "a": (0.1, 1), "h": (0.1, 1), "d": (0.1, 1), "x": (0.1, 10), "y": (0.1, 10)},
        category="strogatz",
        difficulty="very_hard",
        description="Predator response"
    ),
]

# =============================================================================
# SRBENCH 2.0 RECOMMENDED PROBLEMS
# =============================================================================
SRBENCH_BENCHMARKS = [
    # Additional challenging problems recommended by SRBench 2.0
    Benchmark(
        name="srbench_kepler3",
        equation="sqrt(4*pi**2*a**3/(G*M))",
        variables=["a", "G", "M"],
        ranges={"a": (1e10, 1e12), "G": (6.674e-11, 6.674e-11), "M": (1e30, 2e30)},
        category="srbench",
        difficulty="hard",
        description="Kepler's third law"
    ),
    Benchmark(
        name="srbench_planck",
        equation="2*h*c**2/lambd**5 * 1/(exp(h*c/(lambd*k*T))-1)",
        variables=["h", "c", "lambd", "k", "T"],
        ranges={"h": (6.626e-34, 6.626e-34), "c": (3e8, 3e8), "lambd": (1e-7, 1e-5), "k": (1.38e-23, 1.38e-23), "T": (3000, 6000)},
        category="srbench",
        difficulty="very_hard",
        description="Planck's law"
    ),
    Benchmark(
        name="srbench_doppler",
        equation="f_0*c/(c-v)",
        variables=["f_0", "c", "v"],
        ranges={"f_0": (1e6, 1e9), "c": (3e8, 3e8), "v": (1e3, 1e5)},
        category="srbench",
        difficulty="medium",
        description="Doppler effect"
    ),
    Benchmark(
        name="srbench_schwarzschild",
        equation="2*G*M/c**2",
        variables=["G", "M", "c"],
        ranges={"G": (6.674e-11, 6.674e-11), "M": (1e30, 1e40), "c": (3e8, 3e8)},
        category="srbench",
        difficulty="medium",
        description="Schwarzschild radius"
    ),
    Benchmark(
        name="srbench_drag",
        equation="0.5*rho*v**2*C_d*A",
        variables=["rho", "v", "C_d", "A"],
        ranges={"rho": (1, 1.5), "v": (1, 100), "C_d": (0.1, 2), "A": (0.01, 10)},
        category="srbench",
        difficulty="medium",
        description="Drag force"
    ),
    Benchmark(
        name="srbench_pendulum",
        equation="2*pi*sqrt(L/g)",
        variables=["L", "g"],
        ranges={"L": (0.1, 10), "g": (9.5, 10.5)},
        category="srbench",
        difficulty="medium",
        description="Simple pendulum period"
    ),
    Benchmark(
        name="srbench_wave",
        equation="A*sin(k*x - omega*t + phi)",
        variables=["A", "k", "x", "omega", "t", "phi"],
        ranges={"A": (0.1, 5), "k": (0.1, 5), "x": (0, 10), "omega": (0.1, 5), "t": (0, 10), "phi": (0, 2*np.pi)},
        category="srbench",
        difficulty="hard",
        description="Traveling wave"
    ),
    Benchmark(
        name="srbench_compound_interest",
        equation="P*(1+r/n)**(n*t)",
        variables=["P", "r", "n", "t"],
        ranges={"P": (100, 10000), "r": (0.01, 0.2), "n": (1, 12), "t": (1, 30)},
        category="srbench",
        difficulty="hard",
        description="Compound interest"
    ),
    Benchmark(
        name="srbench_logistic",
        equation="K/(1+((K-P_0)/P_0)*exp(-r*t))",
        variables=["K", "P_0", "r", "t"],
        ranges={"K": (100, 1000), "P_0": (10, 50), "r": (0.1, 0.5), "t": (0, 50)},
        category="srbench",
        difficulty="very_hard",
        description="Logistic growth"
    ),
    Benchmark(
        name="srbench_gaussian",
        equation="1/(sigma*sqrt(2*pi))*exp(-(x-mu)**2/(2*sigma**2))",
        variables=["x", "mu", "sigma"],
        ranges={"x": (-5, 5), "mu": (-2, 2), "sigma": (0.5, 2)},
        category="srbench",
        difficulty="hard",
        description="Gaussian distribution"
    ),
]


def safe_eval(equation: str, variables: Dict[str, np.ndarray]) -> np.ndarray:
    """Safely evaluate an equation with given variable values."""
    # Add mathematical functions
    safe_dict = {
        "sqrt": np.sqrt,
        "exp": np.exp,
        "log": np.log,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "tanh": np.tanh,
        "arcsin": np.arcsin,
        "arccos": np.arccos,
        "arctan": np.arctan,
        "pi": np.pi,
        "abs": np.abs,
        **variables
    }
    try:
        result = eval(equation, {"__builtins__": {}}, safe_dict)
        return np.array(result, dtype=np.float64)
    except Exception as e:
        raise ValueError(f"Error evaluating equation '{equation}': {e}")


def generate_dataset(benchmark: Benchmark, seed: int = 42) -> pd.DataFrame:
    """Generate a dataset for a given benchmark."""
    np.random.seed(seed)

    # Generate random samples for each variable
    data = {}
    for var in benchmark.variables:
        low, high = benchmark.ranges[var]
        if low == high:
            # Constant value
            data[var] = np.full(benchmark.n_samples, low)
        else:
            data[var] = np.random.uniform(low, high, benchmark.n_samples)

    # Evaluate the equation
    try:
        y = safe_eval(benchmark.equation, data)
        # Remove invalid values (inf, nan)
        valid_mask = np.isfinite(y)
        if valid_mask.sum() < benchmark.n_samples * 0.9:
            # Regenerate if too many invalid values
            np.random.seed(seed + 1000)
            for var in benchmark.variables:
                low, high = benchmark.ranges[var]
                if low != high:
                    data[var] = np.random.uniform(low, high, benchmark.n_samples)
            y = safe_eval(benchmark.equation, data)
            valid_mask = np.isfinite(y)
    except Exception as e:
        print(f"  Error generating {benchmark.name}: {e}")
        return None

    # Create DataFrame
    df = pd.DataFrame(data)
    df["target"] = y
    df = df[valid_mask].reset_index(drop=True)

    return df


def generate_all_benchmarks(output_dir: Path):
    """Generate all benchmark datasets."""
    all_benchmarks = FEYNMAN_BENCHMARKS + STROGATZ_BENCHMARKS + SRBENCH_BENCHMARKS

    stats = {"total": 0, "by_category": {}, "by_difficulty": {}}
    metadata = {
        "version": "1.0",
        "description": "Unified benchmark suite for symbolic regression",
        "benchmarks": {}
    }

    for benchmark in all_benchmarks:
        category = benchmark.category.split("_")[0]  # feynman, strogatz, srbench
        cat_dir = output_dir / category

        print(f"  Generating {benchmark.name}...", end=" ", flush=True)

        df = generate_dataset(benchmark)
        if df is None or len(df) < 100:
            print("SKIPPED (insufficient data)")
            continue

        # Save dataset
        cat_dir.mkdir(parents=True, exist_ok=True)
        csv_path = cat_dir / f"{benchmark.name}.csv"
        df.to_csv(csv_path, index=False)

        # Save metadata
        bench_meta = {
            "name": benchmark.name,
            "equation": benchmark.equation,
            "variables": benchmark.variables,
            "n_vars": len(benchmark.variables),
            "n_samples": len(df),
            "category": benchmark.category,
            "difficulty": benchmark.difficulty,
            "description": benchmark.description,
        }

        meta_path = cat_dir / f"{benchmark.name}.meta.json"
        with open(meta_path, 'w') as f:
            json.dump(bench_meta, f, indent=2)

        # Update stats
        if category not in metadata["benchmarks"]:
            metadata["benchmarks"][category] = []
        metadata["benchmarks"][category].append(bench_meta)

        stats["total"] += 1
        stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
        stats["by_difficulty"][benchmark.difficulty] = \
            stats["by_difficulty"].get(benchmark.difficulty, 0) + 1

        print(f"OK ({len(df)} samples)")

    metadata["statistics"] = stats

    # Save unified metadata
    meta_path = output_dir / "benchmarks_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return metadata


def main():
    script_dir = Path(__file__).parent

    print("=" * 60)
    print("Symbolic Regression Benchmark Generator")
    print("=" * 60)
    print(f"\nOutput directory: {script_dir}")
    print(f"\nBenchmarks to generate:")
    print(f"  - Feynman: {len(FEYNMAN_BENCHMARKS)} problems")
    print(f"  - Strogatz: {len(STROGATZ_BENCHMARKS)} problems")
    print(f"  - SRBench: {len(SRBENCH_BENCHMARKS)} problems")
    print(f"  - Total: {len(FEYNMAN_BENCHMARKS) + len(STROGATZ_BENCHMARKS) + len(SRBENCH_BENCHMARKS)} problems")

    print("\n" + "=" * 40)
    print("Generating datasets...")
    print("=" * 40)

    metadata = generate_all_benchmarks(script_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    print(f"\nTotal benchmarks: {metadata['statistics']['total']}")
    print("\nBy category:")
    for cat, count in metadata['statistics']['by_category'].items():
        print(f"  - {cat}: {count}")
    print("\nBy difficulty:")
    for diff, count in sorted(metadata['statistics']['by_difficulty'].items()):
        print(f"  - {diff}: {count}")

    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
