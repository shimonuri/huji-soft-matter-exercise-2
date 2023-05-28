import numpy as np
import scipy.special
import scipy.integrate
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 22})
LAMDA_TO_NORMALIZATION = {}


def get_rho(lamda):
    f = lambda theta: np.exp(lamda * scipy.special.legendre(2)(np.cos(theta)))
    if lamda in LAMDA_TO_NORMALIZATION:
        normalization = LAMDA_TO_NORMALIZATION[lamda]
    else:
        normalization = (
            2
            * np.pi
            * scipy.integrate.quad(lambda theta: np.sin(theta) * f(theta), 0, np.pi)[0]
        )
        LAMDA_TO_NORMALIZATION[lamda] = normalization
    return lambda theta: f(theta) / normalization


def find_lamda(mu, should_plot=False):
    function_1 = lambda lamda: mu * lamda
    function_2 = lambda lamda: scipy.integrate.quad(
        lambda theta: (
            2
            * np.pi
            * np.sin(theta)
            * get_rho(lamda)(theta)
            * scipy.special.legendre(2)(np.cos(theta))
        ),
        0,
        np.pi,
    )[0]
    if should_plot:
        _plot(function_1, function_2, mu)
    # find the intersection
    try:
        return scipy.optimize.brentq(
            lambda lamda: function_1(lamda) - function_2(lamda), 0, 20
        )
    except ValueError:
        return 0


def _plot(function_1, function_2, mu):
    # plot the two functions
    plt.plot(
        np.linspace(0, 20, 50), [function_1(lamda) for lamda in np.linspace(0, 20, 50)]
    )
    plt.plot(
        np.linspace(0, 20, 50), [function_2(lamda) for lamda in np.linspace(0, 20, 50)]
    )
    plt.xlabel("lamda")
    plt.ylabel("function")
    plt.title("mu = {}".format(mu))
    plt.show()


def find_s(mu):
    lamda = find_lamda(mu)
    return lamda * mu


def plot_versus_s(mu):
    s_to_integral = lambda s: scipy.integrate.quad(get_rho(mu, s), 0, np.pi)[0]
    plt.scatter(
        np.linspace(0, 20, 100),
        [s_to_integral(s) for s in np.linspace(0, 1, 100)],
    )
    plt.xlabel("s")
    plt.ylabel("integral")
    plt.title("mu = {}".format(mu))
    plt.show()


def plot_s_to_mu(start, end, num):
    plt.scatter(
        np.linspace(start, end, num),
        [find_s(mu) for mu in np.linspace(start, end, num)],
    )
    plt.xlabel("$k_BT/Jq$")
    plt.ylabel("s")
    plt.title("s versus $k_BT/Jq$")
    plt.show()


if __name__ == "__main__":
    plot_s_to_mu(0.06, 0.5, 100)
