"""
Solution for Question 4
"""
import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import pandas
import pathlib
import re
import logging

logging.basicConfig(level=logging.INFO)
plt.rcParams.update({"font.size": 22})

ALPHA_TO_NORMALIZATION = {}


def calculate_free_energies(directory, densities, length, diameter, lamda):
    for density in densities:
        if pathlib.Path(f"{directory}/free_energy_{density}.csv").exists():
            logging.info(f"Skipping density = {density}")
            continue

        logging.info(f"Calculating free energy for density = {density}")
        free_energy = get_free_energy_to_alpha(density, length, diameter, lamda)
        alphas = np.linspace(0, 200, 50)
        free_energies = []
        for alpha in alphas:
            free_energies.append(free_energy(alpha))
            # save to csv
        df = pandas.DataFrame({"alpha": alphas, "free_energy": free_energies})
        df.to_csv(f"{directory}/free_energy_{density}.csv", index=False)
        del df
        logging.info(f"Finished calculating free energy for density = {density}")


def plot_free_energies(directory):
    for file in pathlib.Path(directory).glob("*.csv"):
        df = pandas.read_csv(file)
        density = float(re.search(r"\d+\.\d+", file.name).group(0))
        plt.plot(df["alpha"], df["free_energy"], label=f"density = {density:.2f}")

    plt.xlabel("$\\alpha$")
    plt.ylabel("$\\mathcal{F}(\\alpha)$")
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
    calculate_free_energies('data', np.linspace(0, 50, 2), 1, 1, 1)
    plot_free_energies("data")
