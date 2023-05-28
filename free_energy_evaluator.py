"""
Solution for Question 4
"""
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 22})

ALPHA_TO_NORMALIZATION = {}


def plot_free_energy(number_of_rods, length, diameter, angular_distribution):
    pass


def get_angular_distribution(alpha):
    f = lambda theta: np.cosh(alpha * scipy.special.legendre(2)(np.cos(theta)))
    if alpha in ALPHA_TO_NORMALIZATION:
        normalization = ALPHA_TO_NORMALIZATION[alpha]
    else:
        normalization = (
            2
            * np.pi
            * scipy.integrate.quad(lambda theta: np.sin(theta) * f(theta), 0, np.pi)[0]
        )
        ALPHA_TO_NORMALIZATION[alpha] = normalization
    return lambda theta: f(theta) / normalization
