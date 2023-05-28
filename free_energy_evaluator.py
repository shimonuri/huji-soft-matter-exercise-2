"""
Solution for Question 4
"""
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 22})

ALPHA_TO_NORMALIZATION = {}


def plot_free_energy(densities, length, diameter, lamda):
    for density in densities:
        free_energy = get_free_energy_to_alpha(density, length, diameter, lamda)
        alphas = np.linspace(0, 20, 100)
        free_energies = []
        for alpha in alphas:
            free_energies.append(free_energy(alpha))
        plt.scatter(
            alphas,
            [free_energy(alpha) for alpha in alphas],
            label=f"Density = {density}",
        )
    plt.legend()
    plt.show()


def get_unit_vector(theta, phi):
    return np.array(
        [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta)]
    )


def get_free_energy_to_alpha(density, length, diameter, lamda):
    def first_part(alpha):
        angular_distribution = get_angular_distribution(alpha)
        return (
            2
            * np.pi
            * scipy.integrate.quad(
                lambda theta: angular_distribution(theta)
                * np.sin(theta)
                * (np.log(lamda**3 * angular_distribution(theta)) - 1),
                0,
                np.pi,
            )[0]
        )

    def second_part(alpha):
        angular_distribution = get_angular_distribution(alpha)

        def integrand(theta_1, theta_2):
            sin_angle = np.abs(np.sin(np.abs(theta_1 - theta_2)))
            return (
                np.sin(theta_1)
                * np.sin(theta_2)
                * angular_distribution(theta_1)
                * angular_distribution(theta_2)
                * sin_angle
            )

        return (
            4
            * np.pi**2
            * scipy.integrate.nquad(
                integrand,
                [(0, np.pi), (0, np.pi)],
            )[0]
        )

    return lambda alpha: first_part(
        alpha
    ) + density * diameter * length**2 * second_part(alpha)


def get_angular_distribution(alpha):
    angular_distribution = lambda theta: np.cosh(alpha * np.cos(theta))
    if alpha in ALPHA_TO_NORMALIZATION:
        normalization = ALPHA_TO_NORMALIZATION[alpha]
    else:
        normalization = (
            2
            * np.pi
            * scipy.integrate.quad(
                lambda theta: np.sin(theta) * angular_distribution(theta), 0, np.pi
            )[0]
        )
        ALPHA_TO_NORMALIZATION[alpha] = normalization
    return lambda theta: angular_distribution(theta) / normalization


if __name__ == "__main__":
    plot_free_energy([0, 1, 2, 3, 50], 1, 0.1, 1)
