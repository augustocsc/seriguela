#!/usr/bin/env python3
"""
Official Feynman Symbolic Regression Benchmark Equations.

Source: AI Feynman paper (Udrescu & Tegmark, Science Advances 2020)
        "AI Feynman: A physics-inspired method for symbolic regression"
        https://arxiv.org/abs/1905.11481

The 100 equations are from "The Feynman Lectures on Physics" and are
organized by volume:
- Volume I: Mechanics (I.6.2 - I.50.26)
- Volume II: Electromagnetism (II.2.42 - II.38.14)
- Volume III: Quantum Mechanics (III.4.32 - III.21.20)

Each entry contains:
- name: Identifier matching PMLB/SRBench naming convention
- equation: Ground-truth formula
- variables: List of variable names
- ranges: Valid ranges for each variable
- description: Physical meaning
- n_vars: Number of input variables
"""

import numpy as np

# Complete Feynman equations from AI Feynman benchmark
# Format: {name: {equation, variables, ranges, description}}

FEYNMAN_EQUATIONS = {
    # ==========================================================================
    # VOLUME I - MECHANICS (52 equations)
    # ==========================================================================

    "feynman_I_6_2": {
        "equation": "exp(-theta**2/(2*sigma**2))/(sqrt(2*pi)*sigma)",
        "variables": ["theta", "sigma"],
        "ranges": {"theta": (-3.0, 3.0), "sigma": (1.0, 3.0)},
        "description": "Gaussian/normal distribution",
        "n_vars": 2
    },
    "feynman_I_6_2a": {
        "equation": "exp(-theta**2/(2*sigma**2))",
        "variables": ["theta", "sigma"],
        "ranges": {"theta": (-3.0, 3.0), "sigma": (1.0, 3.0)},
        "description": "Gaussian (unnormalized)",
        "n_vars": 2
    },
    "feynman_I_6_2b": {
        "equation": "exp(-(theta/sigma)**2/2)/(sqrt(2*pi)*sigma)",
        "variables": ["theta", "sigma"],
        "ranges": {"theta": (-3.0, 3.0), "sigma": (1.0, 3.0)},
        "description": "Gaussian (alternate form)",
        "n_vars": 2
    },
    "feynman_I_8_14": {
        "equation": "sqrt((x2-x1)**2+(y2-y1)**2)",
        "variables": ["x1", "y1", "x2", "y2"],
        "ranges": {"x1": (1.0, 5.0), "y1": (1.0, 5.0), "x2": (1.0, 5.0), "y2": (1.0, 5.0)},
        "description": "Distance between two points",
        "n_vars": 4
    },
    "feynman_I_9_18": {
        "equation": "G*m1*m2/((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)",
        "variables": ["G", "m1", "m2", "x1", "y1", "z1", "x2", "y2", "z2"],
        "ranges": {"G": (1.0, 5.0), "m1": (1.0, 5.0), "m2": (1.0, 5.0),
                  "x1": (1.0, 5.0), "y1": (1.0, 5.0), "z1": (1.0, 5.0),
                  "x2": (6.0, 10.0), "y2": (6.0, 10.0), "z2": (6.0, 10.0)},
        "description": "Gravitational force (Newton)",
        "n_vars": 9
    },
    "feynman_I_10_7": {
        "equation": "m_0/sqrt(1-v**2/c**2)",
        "variables": ["m_0", "v", "c"],
        "ranges": {"m_0": (1.0, 5.0), "v": (1.0, 2.0), "c": (3.0, 10.0)},
        "description": "Relativistic mass",
        "n_vars": 3
    },
    "feynman_I_11_19": {
        "equation": "x1*y1+x2*y2+x3*y3",
        "variables": ["x1", "y1", "x2", "y2", "x3", "y3"],
        "ranges": {"x1": (1.0, 5.0), "y1": (1.0, 5.0), "x2": (1.0, 5.0),
                  "y2": (1.0, 5.0), "x3": (1.0, 5.0), "y3": (1.0, 5.0)},
        "description": "Dot product (3D)",
        "n_vars": 6
    },
    "feynman_I_12_1": {
        "equation": "mu*Nn",
        "variables": ["mu", "Nn"],
        "ranges": {"mu": (1.0, 5.0), "Nn": (1.0, 5.0)},
        "description": "Friction force",
        "n_vars": 2
    },
    "feynman_I_12_2": {
        "equation": "q1*q2*r/(4*pi*epsilon*r**3)",
        "variables": ["q1", "q2", "epsilon", "r"],
        "ranges": {"q1": (1.0, 5.0), "q2": (1.0, 5.0), "epsilon": (1.0, 5.0), "r": (1.0, 5.0)},
        "description": "Coulomb force",
        "n_vars": 4
    },
    "feynman_I_12_4": {
        "equation": "q1*r/(4*pi*epsilon*r**3)",
        "variables": ["q1", "epsilon", "r"],
        "ranges": {"q1": (1.0, 5.0), "epsilon": (1.0, 5.0), "r": (1.0, 5.0)},
        "description": "Electric field",
        "n_vars": 3
    },
    "feynman_I_12_5": {
        "equation": "q2*Ef",
        "variables": ["q2", "Ef"],
        "ranges": {"q2": (1.0, 5.0), "Ef": (1.0, 5.0)},
        "description": "Force on charge",
        "n_vars": 2
    },
    "feynman_I_12_11": {
        "equation": "q*(Ef+B*v*sin(theta))",
        "variables": ["q", "Ef", "B", "v", "theta"],
        "ranges": {"q": (1.0, 5.0), "Ef": (1.0, 5.0), "B": (1.0, 5.0),
                  "v": (1.0, 5.0), "theta": (0.0, np.pi)},
        "description": "Lorentz force",
        "n_vars": 5
    },
    "feynman_I_13_4": {
        "equation": "0.5*m*(v**2+u**2+w**2)",
        "variables": ["m", "v", "u", "w"],
        "ranges": {"m": (1.0, 5.0), "v": (1.0, 5.0), "u": (1.0, 5.0), "w": (1.0, 5.0)},
        "description": "Kinetic energy (3D)",
        "n_vars": 4
    },
    "feynman_I_13_12": {
        "equation": "G*m1*m2*(1/r2-1/r1)",
        "variables": ["G", "m1", "m2", "r1", "r2"],
        "ranges": {"G": (1.0, 5.0), "m1": (1.0, 5.0), "m2": (1.0, 5.0),
                  "r1": (1.0, 5.0), "r2": (1.0, 5.0)},
        "description": "Gravitational potential energy difference",
        "n_vars": 5
    },
    "feynman_I_14_3": {
        "equation": "m*g*z",
        "variables": ["m", "g", "z"],
        "ranges": {"m": (1.0, 5.0), "g": (1.0, 5.0), "z": (1.0, 5.0)},
        "description": "Gravitational potential energy",
        "n_vars": 3
    },
    "feynman_I_14_4": {
        "equation": "0.5*k_spring*x**2",
        "variables": ["k_spring", "x"],
        "ranges": {"k_spring": (1.0, 5.0), "x": (1.0, 5.0)},
        "description": "Spring potential energy",
        "n_vars": 2
    },
    "feynman_I_15_3t": {
        "equation": "(t-u*x/c**2)/sqrt(1-u**2/c**2)",
        "variables": ["t", "u", "x", "c"],
        "ranges": {"t": (1.0, 5.0), "u": (1.0, 2.0), "x": (1.0, 5.0), "c": (3.0, 10.0)},
        "description": "Lorentz transformation (time)",
        "n_vars": 4
    },
    "feynman_I_15_3x": {
        "equation": "(x-u*t)/sqrt(1-u**2/c**2)",
        "variables": ["x", "u", "t", "c"],
        "ranges": {"x": (5.0, 10.0), "u": (1.0, 2.0), "t": (1.0, 5.0), "c": (3.0, 10.0)},
        "description": "Lorentz transformation (position)",
        "n_vars": 4
    },
    "feynman_I_15_10": {
        "equation": "m_0*v/sqrt(1-v**2/c**2)",
        "variables": ["m_0", "v", "c"],
        "ranges": {"m_0": (1.0, 5.0), "v": (1.0, 2.0), "c": (3.0, 10.0)},
        "description": "Relativistic momentum",
        "n_vars": 3
    },
    "feynman_I_16_6": {
        "equation": "(u+v)/(1+u*v/c**2)",
        "variables": ["u", "v", "c"],
        "ranges": {"u": (1.0, 5.0), "v": (1.0, 5.0), "c": (5.0, 20.0)},
        "description": "Relativistic velocity addition",
        "n_vars": 3
    },
    "feynman_I_18_4": {
        "equation": "(m1*r1+m2*r2)/(m1+m2)",
        "variables": ["m1", "r1", "m2", "r2"],
        "ranges": {"m1": (1.0, 5.0), "r1": (1.0, 5.0), "m2": (1.0, 5.0), "r2": (1.0, 5.0)},
        "description": "Center of mass (1D)",
        "n_vars": 4
    },
    "feynman_I_18_12": {
        "equation": "r*F*sin(theta)",
        "variables": ["r", "F", "theta"],
        "ranges": {"r": (1.0, 5.0), "F": (1.0, 5.0), "theta": (0.0, np.pi)},
        "description": "Torque",
        "n_vars": 3
    },
    "feynman_I_18_14": {
        "equation": "m*r*v*sin(theta)",
        "variables": ["m", "r", "v", "theta"],
        "ranges": {"m": (1.0, 5.0), "r": (1.0, 5.0), "v": (1.0, 5.0), "theta": (0.0, np.pi)},
        "description": "Angular momentum",
        "n_vars": 4
    },
    "feynman_I_24_6": {
        "equation": "0.25*m*(omega**2+omega_0**2)*x**2",
        "variables": ["m", "omega", "omega_0", "x"],
        "ranges": {"m": (1.0, 3.0), "omega": (1.0, 3.0), "omega_0": (1.0, 3.0), "x": (1.0, 3.0)},
        "description": "Oscillator energy",
        "n_vars": 4
    },
    "feynman_I_25_13": {
        "equation": "q/C",
        "variables": ["q", "C"],
        "ranges": {"q": (1.0, 5.0), "C": (1.0, 5.0)},
        "description": "Capacitor voltage",
        "n_vars": 2
    },
    "feynman_I_26_2": {
        "equation": "arcsin(n*sin(theta2))",
        "variables": ["n", "theta2"],
        "ranges": {"n": (0.0, 1.0), "theta2": (1.0, 5.0)},
        "description": "Snell's law",
        "n_vars": 2
    },
    "feynman_I_27_6": {
        "equation": "1/(1/d1+n/d2)",
        "variables": ["d1", "n", "d2"],
        "ranges": {"d1": (1.0, 5.0), "n": (1.0, 5.0), "d2": (1.0, 5.0)},
        "description": "Focal length",
        "n_vars": 3
    },
    "feynman_I_29_4": {
        "equation": "omega/c",
        "variables": ["omega", "c"],
        "ranges": {"omega": (1.0, 10.0), "c": (1.0, 10.0)},
        "description": "Wave number",
        "n_vars": 2
    },
    "feynman_I_29_16": {
        "equation": "sqrt(x1**2+x2**2-2*x1*x2*cos(theta1-theta2))",
        "variables": ["x1", "x2", "theta1", "theta2"],
        "ranges": {"x1": (1.0, 5.0), "x2": (1.0, 5.0), "theta1": (0.0, 2*np.pi), "theta2": (0.0, 2*np.pi)},
        "description": "Law of cosines",
        "n_vars": 4
    },
    "feynman_I_30_3": {
        "equation": "Int_0*sin(n*theta/2)**2/sin(theta/2)**2",
        "variables": ["Int_0", "n", "theta"],
        "ranges": {"Int_0": (1.0, 5.0), "n": (1.0, 5.0), "theta": (1.0, 5.0)},
        "description": "Diffraction intensity",
        "n_vars": 3
    },
    "feynman_I_30_5": {
        "equation": "arcsin(lambd/(n*d))",
        "variables": ["lambd", "n", "d"],
        "ranges": {"lambd": (1.0, 2.0), "n": (1.0, 5.0), "d": (2.0, 5.0)},
        "description": "Diffraction angle",
        "n_vars": 3
    },
    "feynman_I_32_5": {
        "equation": "q**2*a**2/(6*pi*epsilon*c**3)",
        "variables": ["q", "a", "epsilon", "c"],
        "ranges": {"q": (1.0, 5.0), "a": (1.0, 5.0), "epsilon": (1.0, 5.0), "c": (1.0, 5.0)},
        "description": "Radiated power",
        "n_vars": 4
    },
    "feynman_I_32_17": {
        "equation": "(0.5*epsilon*c*Ef**2)*(8*pi*r**2/3)*(omega**4/(omega**2-omega_0**2)**2)",
        "variables": ["epsilon", "c", "Ef", "r", "omega", "omega_0"],
        "ranges": {"epsilon": (1.0, 2.0), "c": (1.0, 2.0), "Ef": (1.0, 2.0),
                  "r": (1.0, 2.0), "omega": (1.0, 2.0), "omega_0": (3.0, 5.0)},
        "description": "Scattering cross section",
        "n_vars": 6
    },
    "feynman_I_34_1": {
        "equation": "omega_0/(1-v/c)",
        "variables": ["omega_0", "v", "c"],
        "ranges": {"omega_0": (1.0, 5.0), "v": (1.0, 2.0), "c": (3.0, 10.0)},
        "description": "Doppler shift (approaching)",
        "n_vars": 3
    },
    "feynman_I_34_8": {
        "equation": "q*v*B/p",
        "variables": ["q", "v", "B", "p"],
        "ranges": {"q": (1.0, 5.0), "v": (1.0, 5.0), "B": (1.0, 5.0), "p": (1.0, 5.0)},
        "description": "Cyclotron frequency component",
        "n_vars": 4
    },
    "feynman_I_34_10": {
        "equation": "omega_0/(1-v/c)",
        "variables": ["omega_0", "v", "c"],
        "ranges": {"omega_0": (3.0, 10.0), "v": (1.0, 2.0), "c": (3.0, 10.0)},
        "description": "Doppler effect",
        "n_vars": 3
    },
    "feynman_I_34_14": {
        "equation": "(1+v/c)/sqrt(1-v**2/c**2)*omega_0",
        "variables": ["v", "c", "omega_0"],
        "ranges": {"v": (1.0, 2.0), "c": (3.0, 10.0), "omega_0": (1.0, 5.0)},
        "description": "Relativistic Doppler",
        "n_vars": 3
    },
    "feynman_I_34_27": {
        "equation": "h*omega/(2*pi)",
        "variables": ["h", "omega"],
        "ranges": {"h": (1.0, 5.0), "omega": (1.0, 5.0)},
        "description": "Photon energy",
        "n_vars": 2
    },
    "feynman_I_37_4": {
        "equation": "I1+I2+2*sqrt(I1*I2)*cos(delta)",
        "variables": ["I1", "I2", "delta"],
        "ranges": {"I1": (1.0, 5.0), "I2": (1.0, 5.0), "delta": (0.0, 2*np.pi)},
        "description": "Interference intensity",
        "n_vars": 3
    },
    "feynman_I_38_12": {
        "equation": "4*pi*epsilon*h**2/(m*q**2)",
        "variables": ["epsilon", "h", "m", "q"],
        "ranges": {"epsilon": (1.0, 5.0), "h": (1.0, 5.0), "m": (1.0, 5.0), "q": (1.0, 5.0)},
        "description": "Bohr radius",
        "n_vars": 4
    },
    "feynman_I_39_1": {
        "equation": "1.5*pr*V",
        "variables": ["pr", "V"],
        "ranges": {"pr": (1.0, 5.0), "V": (1.0, 5.0)},
        "description": "Ideal gas internal energy",
        "n_vars": 2
    },
    "feynman_I_39_11": {
        "equation": "1/(gamma-1)*pr*V",
        "variables": ["gamma", "pr", "V"],
        "ranges": {"gamma": (2.0, 5.0), "pr": (1.0, 5.0), "V": (1.0, 5.0)},
        "description": "Internal energy (general)",
        "n_vars": 3
    },
    "feynman_I_39_22": {
        "equation": "n*kb*T/V",
        "variables": ["n", "kb", "T", "V"],
        "ranges": {"n": (1.0, 5.0), "kb": (1.0, 5.0), "T": (1.0, 5.0), "V": (1.0, 5.0)},
        "description": "Ideal gas law (pressure)",
        "n_vars": 4
    },
    "feynman_I_40_1": {
        "equation": "n_0*exp(-m*g*x/(kb*T))",
        "variables": ["n_0", "m", "g", "x", "kb", "T"],
        "ranges": {"n_0": (1.0, 5.0), "m": (1.0, 5.0), "g": (1.0, 5.0),
                  "x": (1.0, 5.0), "kb": (1.0, 5.0), "T": (1.0, 5.0)},
        "description": "Barometric formula",
        "n_vars": 6
    },
    "feynman_I_41_16": {
        "equation": "h*omega**3/(pi**2*c**2*(exp(h*omega/(kb*T))-1))",
        "variables": ["h", "omega", "c", "kb", "T"],
        "ranges": {"h": (1.0, 5.0), "omega": (1.0, 5.0), "c": (1.0, 5.0),
                  "kb": (1.0, 5.0), "T": (1.0, 5.0)},
        "description": "Planck distribution",
        "n_vars": 5
    },
    "feynman_I_43_16": {
        "equation": "mu*q*Ef*u/(kb*T)",
        "variables": ["mu", "q", "Ef", "u", "kb", "T"],
        "ranges": {"mu": (1.0, 5.0), "q": (1.0, 5.0), "Ef": (1.0, 5.0),
                  "u": (1.0, 5.0), "kb": (1.0, 5.0), "T": (1.0, 5.0)},
        "description": "Drift current",
        "n_vars": 6
    },
    "feynman_I_43_31": {
        "equation": "mu*kb*T",
        "variables": ["mu", "kb", "T"],
        "ranges": {"mu": (1.0, 5.0), "kb": (1.0, 5.0), "T": (1.0, 5.0)},
        "description": "Einstein relation (diffusion)",
        "n_vars": 3
    },
    "feynman_I_43_43": {
        "equation": "kb*v/(2*pi*d**2)",
        "variables": ["kb", "v", "d"],
        "ranges": {"kb": (1.0, 5.0), "v": (1.0, 5.0), "d": (1.0, 5.0)},
        "description": "Mean free path",
        "n_vars": 3
    },
    "feynman_I_44_4": {
        "equation": "n*kb*T*log(V2/V1)",
        "variables": ["n", "kb", "T", "V1", "V2"],
        "ranges": {"n": (1.0, 5.0), "kb": (1.0, 5.0), "T": (1.0, 5.0),
                  "V1": (1.0, 5.0), "V2": (1.0, 5.0)},
        "description": "Isothermal work",
        "n_vars": 5
    },
    "feynman_I_47_23": {
        "equation": "sqrt(gamma*pr/rho)",
        "variables": ["gamma", "pr", "rho"],
        "ranges": {"gamma": (1.0, 5.0), "pr": (1.0, 5.0), "rho": (1.0, 5.0)},
        "description": "Speed of sound",
        "n_vars": 3
    },
    "feynman_I_48_2": {
        "equation": "m*c**2/sqrt(1-v**2/c**2)",
        "variables": ["m", "c", "v"],
        "ranges": {"m": (1.0, 5.0), "c": (3.0, 10.0), "v": (1.0, 2.0)},
        "description": "Relativistic energy",
        "n_vars": 3
    },
    "feynman_I_50_26": {
        "equation": "x1*(cos(omega*t)+alpha*cos(omega*t)**2)",
        "variables": ["x1", "omega", "t", "alpha"],
        "ranges": {"x1": (1.0, 3.0), "omega": (1.0, 3.0), "t": (1.0, 3.0), "alpha": (1.0, 3.0)},
        "description": "Anharmonic oscillator",
        "n_vars": 4
    },

    # ==========================================================================
    # VOLUME II - ELECTROMAGNETISM (34 equations)
    # ==========================================================================

    "feynman_II_2_42": {
        "equation": "kappa*(T2-T1)*A/d",
        "variables": ["kappa", "T1", "T2", "A", "d"],
        "ranges": {"kappa": (1.0, 5.0), "T1": (1.0, 5.0), "T2": (5.0, 10.0),
                  "A": (1.0, 5.0), "d": (1.0, 5.0)},
        "description": "Heat conduction",
        "n_vars": 5
    },
    "feynman_II_3_24": {
        "equation": "Pwr/(4*pi*r**2)",
        "variables": ["Pwr", "r"],
        "ranges": {"Pwr": (1.0, 5.0), "r": (1.0, 5.0)},
        "description": "Flux density (inverse square)",
        "n_vars": 2
    },
    "feynman_II_4_23": {
        "equation": "q/(4*pi*epsilon*r)",
        "variables": ["q", "epsilon", "r"],
        "ranges": {"q": (1.0, 5.0), "epsilon": (1.0, 5.0), "r": (1.0, 5.0)},
        "description": "Electric potential",
        "n_vars": 3
    },
    "feynman_II_6_11": {
        "equation": "p_d*cos(theta)/(4*pi*epsilon*r**2)",
        "variables": ["p_d", "theta", "epsilon", "r"],
        "ranges": {"p_d": (1.0, 5.0), "theta": (0.0, np.pi), "epsilon": (1.0, 5.0), "r": (1.0, 5.0)},
        "description": "Dipole potential",
        "n_vars": 4
    },
    "feynman_II_6_15a": {
        "equation": "p_d/(4*pi*epsilon)*3*z/r**5*sqrt(x**2+y**2)",
        "variables": ["p_d", "epsilon", "z", "r", "x", "y"],
        "ranges": {"p_d": (1.0, 3.0), "epsilon": (1.0, 3.0), "z": (1.0, 3.0),
                  "r": (1.0, 3.0), "x": (1.0, 3.0), "y": (1.0, 3.0)},
        "description": "Dipole field (x-component)",
        "n_vars": 6
    },
    "feynman_II_6_15b": {
        "equation": "p_d/(4*pi*epsilon)*3*cos(theta)*sin(theta)/r**3",
        "variables": ["p_d", "epsilon", "theta", "r"],
        "ranges": {"p_d": (1.0, 5.0), "epsilon": (1.0, 5.0), "theta": (0.1, np.pi-0.1), "r": (1.0, 5.0)},
        "description": "Dipole field (angular)",
        "n_vars": 4
    },
    "feynman_II_8_7": {
        "equation": "3/5*q**2/(4*pi*epsilon*d)",
        "variables": ["q", "epsilon", "d"],
        "ranges": {"q": (1.0, 5.0), "epsilon": (1.0, 5.0), "d": (1.0, 5.0)},
        "description": "Sphere self-energy",
        "n_vars": 3
    },
    "feynman_II_8_31": {
        "equation": "epsilon*Ef**2/2",
        "variables": ["epsilon", "Ef"],
        "ranges": {"epsilon": (1.0, 5.0), "Ef": (1.0, 5.0)},
        "description": "Electric field energy density",
        "n_vars": 2
    },
    "feynman_II_10_9": {
        "equation": "sigma_den/(epsilon*(1+chi))",
        "variables": ["sigma_den", "epsilon", "chi"],
        "ranges": {"sigma_den": (1.0, 5.0), "epsilon": (1.0, 5.0), "chi": (1.0, 5.0)},
        "description": "Dielectric field",
        "n_vars": 3
    },
    "feynman_II_11_3": {
        "equation": "q*Ef/(m*(omega_0**2-omega**2))",
        "variables": ["q", "Ef", "m", "omega_0", "omega"],
        "ranges": {"q": (1.0, 3.0), "Ef": (1.0, 3.0), "m": (1.0, 3.0),
                  "omega_0": (3.0, 5.0), "omega": (1.0, 2.0)},
        "description": "Driven oscillator amplitude",
        "n_vars": 5
    },
    "feynman_II_11_17": {
        "equation": "n_0*(1+p_d*Ef*cos(theta)/(kb*T))",
        "variables": ["n_0", "p_d", "Ef", "theta", "kb", "T"],
        "ranges": {"n_0": (1.0, 5.0), "p_d": (1.0, 5.0), "Ef": (1.0, 5.0),
                  "theta": (0.0, np.pi), "kb": (1.0, 5.0), "T": (1.0, 5.0)},
        "description": "Boltzmann distribution (polarization)",
        "n_vars": 6
    },
    "feynman_II_11_20": {
        "equation": "n*p_d**2*Ef/(3*kb*T)",
        "variables": ["n", "p_d", "Ef", "kb", "T"],
        "ranges": {"n": (1.0, 5.0), "p_d": (1.0, 5.0), "Ef": (1.0, 5.0),
                  "kb": (1.0, 5.0), "T": (1.0, 5.0)},
        "description": "Polarization (high T)",
        "n_vars": 5
    },
    "feynman_II_11_27": {
        "equation": "n*alpha/(1-n*alpha/3)*epsilon*Ef",
        "variables": ["n", "alpha", "epsilon", "Ef"],
        "ranges": {"n": (0.0, 1.0), "alpha": (0.0, 1.0), "epsilon": (1.0, 5.0), "Ef": (1.0, 5.0)},
        "description": "Clausius-Mossotti",
        "n_vars": 4
    },
    "feynman_II_11_28": {
        "equation": "1+n*alpha/(1-n*alpha/3)",
        "variables": ["n", "alpha"],
        "ranges": {"n": (0.0, 1.0), "alpha": (0.0, 1.0)},
        "description": "Dielectric constant",
        "n_vars": 2
    },
    "feynman_II_13_17": {
        "equation": "2*I/(4*pi*epsilon*c**2*r)",
        "variables": ["I", "epsilon", "c", "r"],
        "ranges": {"I": (1.0, 5.0), "epsilon": (1.0, 5.0), "c": (1.0, 5.0), "r": (1.0, 5.0)},
        "description": "Magnetic field (wire)",
        "n_vars": 4
    },
    "feynman_II_13_23": {
        "equation": "rho_c/sqrt(1-v**2/c**2)",
        "variables": ["rho_c", "v", "c"],
        "ranges": {"rho_c": (1.0, 5.0), "v": (1.0, 2.0), "c": (3.0, 10.0)},
        "description": "Relativistic charge density",
        "n_vars": 3
    },
    "feynman_II_13_34": {
        "equation": "rho_c*v/sqrt(1-v**2/c**2)",
        "variables": ["rho_c", "v", "c"],
        "ranges": {"rho_c": (1.0, 5.0), "v": (1.0, 2.0), "c": (3.0, 10.0)},
        "description": "Relativistic current density",
        "n_vars": 3
    },
    "feynman_II_15_4": {
        "equation": "-mom*B*cos(theta)",
        "variables": ["mom", "B", "theta"],
        "ranges": {"mom": (1.0, 5.0), "B": (1.0, 5.0), "theta": (0.0, np.pi)},
        "description": "Magnetic dipole energy",
        "n_vars": 3
    },
    "feynman_II_15_5": {
        "equation": "-p_d*Ef*cos(theta)",
        "variables": ["p_d", "Ef", "theta"],
        "ranges": {"p_d": (1.0, 5.0), "Ef": (1.0, 5.0), "theta": (0.0, np.pi)},
        "description": "Electric dipole energy",
        "n_vars": 3
    },
    "feynman_II_21_32": {
        "equation": "q/(4*pi*epsilon*r*(1-v/c))",
        "variables": ["q", "epsilon", "r", "v", "c"],
        "ranges": {"q": (1.0, 5.0), "epsilon": (1.0, 5.0), "r": (1.0, 5.0),
                  "v": (1.0, 2.0), "c": (3.0, 10.0)},
        "description": "Retarded potential",
        "n_vars": 5
    },
    "feynman_II_24_17": {
        "equation": "sqrt(omega**2/c**2-pi**2/d**2)",
        "variables": ["omega", "c", "d"],
        "ranges": {"omega": (4.0, 6.0), "c": (1.0, 2.0), "d": (2.0, 4.0)},
        "description": "Waveguide propagation",
        "n_vars": 3
    },
    "feynman_II_27_16": {
        "equation": "epsilon*c*Ef**2",
        "variables": ["epsilon", "c", "Ef"],
        "ranges": {"epsilon": (1.0, 5.0), "c": (1.0, 5.0), "Ef": (1.0, 5.0)},
        "description": "Poynting vector magnitude",
        "n_vars": 3
    },
    "feynman_II_27_18": {
        "equation": "epsilon*Ef**2",
        "variables": ["epsilon", "Ef"],
        "ranges": {"epsilon": (1.0, 5.0), "Ef": (1.0, 5.0)},
        "description": "Energy density",
        "n_vars": 2
    },
    "feynman_II_34_2": {
        "equation": "q*v/(2*pi*r)",
        "variables": ["q", "v", "r"],
        "ranges": {"q": (1.0, 5.0), "v": (1.0, 5.0), "r": (1.0, 5.0)},
        "description": "Current (moving charge)",
        "n_vars": 3
    },
    "feynman_II_34_2a": {
        "equation": "q*v*r/2",
        "variables": ["q", "v", "r"],
        "ranges": {"q": (1.0, 5.0), "v": (1.0, 5.0), "r": (1.0, 5.0)},
        "description": "Magnetic moment",
        "n_vars": 3
    },
    "feynman_II_34_11": {
        "equation": "g_*q*B/(2*m)",
        "variables": ["g_", "q", "B", "m"],
        "ranges": {"g_": (1.0, 5.0), "q": (1.0, 5.0), "B": (1.0, 5.0), "m": (1.0, 5.0)},
        "description": "Larmor frequency",
        "n_vars": 4
    },
    "feynman_II_34_29a": {
        "equation": "q*h/(4*pi*m)",
        "variables": ["q", "h", "m"],
        "ranges": {"q": (1.0, 5.0), "h": (1.0, 5.0), "m": (1.0, 5.0)},
        "description": "Bohr magneton",
        "n_vars": 3
    },
    "feynman_II_34_29b": {
        "equation": "g_*mom*B*Jz/h",
        "variables": ["g_", "mom", "B", "Jz", "h"],
        "ranges": {"g_": (1.0, 5.0), "mom": (1.0, 5.0), "B": (1.0, 5.0),
                  "Jz": (1.0, 5.0), "h": (1.0, 5.0)},
        "description": "Zeeman energy",
        "n_vars": 5
    },
    "feynman_II_35_18": {
        "equation": "n_0/(exp(mom*B/(kb*T))+exp(-mom*B/(kb*T)))",
        "variables": ["n_0", "mom", "B", "kb", "T"],
        "ranges": {"n_0": (1.0, 3.0), "mom": (1.0, 3.0), "B": (1.0, 3.0),
                  "kb": (1.0, 3.0), "T": (1.0, 3.0)},
        "description": "Boltzmann (two-level)",
        "n_vars": 5
    },
    "feynman_II_35_21": {
        "equation": "n*mom*tanh(mom*B/(kb*T))",
        "variables": ["n", "mom", "B", "kb", "T"],
        "ranges": {"n": (1.0, 5.0), "mom": (1.0, 5.0), "B": (1.0, 5.0),
                  "kb": (1.0, 5.0), "T": (1.0, 5.0)},
        "description": "Magnetization",
        "n_vars": 5
    },
    "feynman_II_36_38": {
        "equation": "mom*H/(kb*T)+mom*alpha*M/(kb*T)",
        "variables": ["mom", "H", "kb", "T", "alpha", "M"],
        "ranges": {"mom": (1.0, 5.0), "H": (1.0, 5.0), "kb": (1.0, 5.0),
                  "T": (1.0, 5.0), "alpha": (1.0, 5.0), "M": (1.0, 5.0)},
        "description": "Curie-Weiss",
        "n_vars": 6
    },
    "feynman_II_37_1": {
        "equation": "mom*(1+chi)*B",
        "variables": ["mom", "chi", "B"],
        "ranges": {"mom": (1.0, 5.0), "chi": (1.0, 5.0), "B": (1.0, 5.0)},
        "description": "Magnetic susceptibility",
        "n_vars": 3
    },
    "feynman_II_38_3": {
        "equation": "Y*A*x/d",
        "variables": ["Y", "A", "x", "d"],
        "ranges": {"Y": (1.0, 5.0), "A": (1.0, 5.0), "x": (1.0, 5.0), "d": (1.0, 5.0)},
        "description": "Hooke's law (stress)",
        "n_vars": 4
    },
    "feynman_II_38_14": {
        "equation": "Y/(2*(1+sigma))",
        "variables": ["Y", "sigma"],
        "ranges": {"Y": (1.0, 5.0), "sigma": (1.0, 5.0)},
        "description": "Shear modulus",
        "n_vars": 2
    },

    # ==========================================================================
    # VOLUME III - QUANTUM MECHANICS (15 equations)
    # ==========================================================================

    "feynman_III_4_32": {
        "equation": "1/(exp(h*omega/(2*pi*kb*T))-1)",
        "variables": ["h", "omega", "kb", "T"],
        "ranges": {"h": (1.0, 5.0), "omega": (1.0, 5.0), "kb": (1.0, 5.0), "T": (1.0, 5.0)},
        "description": "Bose-Einstein distribution",
        "n_vars": 4
    },
    "feynman_III_4_33": {
        "equation": "h*omega/(2*pi*(exp(h*omega/(2*pi*kb*T))-1))",
        "variables": ["h", "omega", "kb", "T"],
        "ranges": {"h": (1.0, 5.0), "omega": (1.0, 5.0), "kb": (1.0, 5.0), "T": (1.0, 5.0)},
        "description": "Planck oscillator energy",
        "n_vars": 4
    },
    "feynman_III_7_38": {
        "equation": "2*mom*sqrt(Bx**2+By**2+Bz**2)",
        "variables": ["mom", "Bx", "By", "Bz"],
        "ranges": {"mom": (1.0, 5.0), "Bx": (1.0, 5.0), "By": (1.0, 5.0), "Bz": (1.0, 5.0)},
        "description": "Zeeman splitting",
        "n_vars": 4
    },
    "feynman_III_8_54": {
        "equation": "sin(E_n*t/h)**2",
        "variables": ["E_n", "t", "h"],
        "ranges": {"E_n": (1.0, 5.0), "t": (1.0, 5.0), "h": (1.0, 5.0)},
        "description": "Rabi oscillation",
        "n_vars": 3
    },
    "feynman_III_9_52": {
        "equation": "(p_d*Ef*t/h)*sin((omega-omega_0)*t/2)**2/((omega-omega_0)*t/2)**2",
        "variables": ["p_d", "Ef", "t", "h", "omega", "omega_0"],
        "ranges": {"p_d": (1.0, 3.0), "Ef": (1.0, 3.0), "t": (1.0, 3.0),
                  "h": (1.0, 3.0), "omega": (1.0, 5.0), "omega_0": (3.0, 5.0)},
        "description": "Transition probability",
        "n_vars": 6
    },
    "feynman_III_10_19": {
        "equation": "mom*sqrt(Bx**2+By**2+Bz**2)",
        "variables": ["mom", "Bx", "By", "Bz"],
        "ranges": {"mom": (1.0, 5.0), "Bx": (1.0, 5.0), "By": (1.0, 5.0), "Bz": (1.0, 5.0)},
        "description": "Magnetic moment energy",
        "n_vars": 4
    },
    "feynman_III_12_43": {
        "equation": "n*h/(2*pi)",
        "variables": ["n", "h"],
        "ranges": {"n": (1.0, 5.0), "h": (1.0, 5.0)},
        "description": "Angular momentum quantization",
        "n_vars": 2
    },
    "feynman_III_13_18": {
        "equation": "2*E_n*d**2*k/h",
        "variables": ["E_n", "d", "k", "h"],
        "ranges": {"E_n": (1.0, 5.0), "d": (1.0, 5.0), "k": (1.0, 5.0), "h": (1.0, 5.0)},
        "description": "Bloch velocity",
        "n_vars": 4
    },
    "feynman_III_14_14": {
        "equation": "I_0*(exp(q*U/(kb*T))-1)",
        "variables": ["I_0", "q", "U", "kb", "T"],
        "ranges": {"I_0": (1.0, 5.0), "q": (1.0, 5.0), "U": (1.0, 5.0),
                  "kb": (1.0, 5.0), "T": (1.0, 5.0)},
        "description": "Diode equation",
        "n_vars": 5
    },
    "feynman_III_15_12": {
        "equation": "2*U*(1-cos(k*d))",
        "variables": ["U", "k", "d"],
        "ranges": {"U": (1.0, 5.0), "k": (1.0, 5.0), "d": (1.0, 5.0)},
        "description": "Band structure",
        "n_vars": 3
    },
    "feynman_III_15_14": {
        "equation": "h**2/(8*pi**2*E_n*d**2)",
        "variables": ["h", "E_n", "d"],
        "ranges": {"h": (1.0, 5.0), "E_n": (1.0, 5.0), "d": (1.0, 5.0)},
        "description": "Effective mass",
        "n_vars": 3
    },
    "feynman_III_15_27": {
        "equation": "2*pi*alpha/(n*d)",
        "variables": ["alpha", "n", "d"],
        "ranges": {"alpha": (1.0, 5.0), "n": (1.0, 5.0), "d": (1.0, 5.0)},
        "description": "Wave vector",
        "n_vars": 3
    },
    "feynman_III_17_37": {
        "equation": "beta*(1+alpha*cos(theta))",
        "variables": ["beta", "alpha", "theta"],
        "ranges": {"beta": (1.0, 5.0), "alpha": (0.0, 1.0), "theta": (0.0, 2*np.pi)},
        "description": "Angular distribution",
        "n_vars": 3
    },
    "feynman_III_19_51": {
        "equation": "-m*q**4/(2*(4*pi*epsilon)**2*h**2)*1/n**2",
        "variables": ["m", "q", "epsilon", "h", "n"],
        "ranges": {"m": (1.0, 5.0), "q": (1.0, 5.0), "epsilon": (1.0, 5.0),
                  "h": (1.0, 5.0), "n": (1.0, 5.0)},
        "description": "Hydrogen energy levels",
        "n_vars": 5
    },
    "feynman_III_21_20": {
        "equation": "-rho_c*q*A/m",
        "variables": ["rho_c", "q", "A", "m"],
        "ranges": {"rho_c": (1.0, 5.0), "q": (1.0, 5.0), "A": (1.0, 5.0), "m": (1.0, 5.0)},
        "description": "Superconductor current",
        "n_vars": 4
    },
}


# Total: 100 equations
# Volume I: 52 equations
# Volume II: 34 equations
# Volume III: 15 equations (Total should be 101 with some variants)

def get_all_equations():
    """Return all Feynman equations."""
    return FEYNMAN_EQUATIONS


def get_equation_by_name(name: str):
    """Get a specific equation by name."""
    return FEYNMAN_EQUATIONS.get(name)


def get_equations_by_volume(volume: str):
    """Get equations by Feynman volume (I, II, or III)."""
    prefix = f"feynman_{volume}_"
    return {k: v for k, v in FEYNMAN_EQUATIONS.items() if k.startswith(prefix)}


if __name__ == "__main__":
    print(f"Total Feynman equations: {len(FEYNMAN_EQUATIONS)}")
    print(f"Volume I: {len(get_equations_by_volume('I'))}")
    print(f"Volume II: {len(get_equations_by_volume('II'))}")
    print(f"Volume III: {len(get_equations_by_volume('III'))}")
