"""
Module of kernel functions
"""

# imports
import numpy as np


# declaration

def indicator(statement):
    if statement:
        return 1
    else:
        return 0


def uniform(u, kernelorder, onlyroughness=False, onlymukk=False):
    """
    Uniform kernel function
    :param u: input
    :param kernelorder: order of the kernel function, 6 or 4 or 2
    :param onlyroughness: if True only the roughness of the kernel is returned, a scalar defined as
    integrate_{-infty}^{intfty} (K(u))^2 du
    :param onlymukk: if True only  mukk := integrate_{-infty}^{intfty} u^kernelorder * K(u) du is returned
    :return: either K(u) or the roughness or mukk
    """

    if kernelorder == 2:
        if onlyroughness:
            # Function to be integrated to get roughness
            def integrand(x):
                return (0.5 * indicator((-1 <= x <= 1))) ** 2
            # Integrate to get roughness
            from scipy.integrate import quad
            r = quad(integrand, -np.inf, np.inf)[0]
            return r
        elif onlymukk:
            # Function to be integrated to get mu2k
            def integrand(x):
                return (0.5 * indicator((-1 <= x <= 1))) * x ** kernelorder
            # Integrate to get roughness
            from scipy.integrate import quad
            mukk = quad(integrand, -np.inf, np.inf)[0]
            return mukk
        else:
            return 0.5 * indicator((-1 <= u <= 1))
    else:
        raise Exception('For uniform kernel only kernelorder=2 is available\n')


def triangular(u, kernelorder, onlyroughness=False, onlymukk=False):
    """
    Triangular kernel function - second order
    :param u: inout to kernel function
    :param kernelorder: order of the kernel function
    :param onlyroughness: if True only the roughness of the kernel is returned, a scalar defined as
    integrate_{-infty}^{intfty} (K(u))^2 du
    :param onlymukk: if True only  mukk := integrate_{-infty}^{intfty} u^kernelorder * K(u) du is returned
    :return: either K(u) or the roughness or mukk
    """

    if kernelorder == 2:
        if onlyroughness:
            # Function to be integrated to get roughness
            def integrand(x):
                return ((1-abs(x))*indicator((-1 <= x <= 1))) ** 2
            # Integrate to get roughness
            from scipy.integrate import quad
            r = quad(integrand, -np.inf, np.inf)[0]
            return r
        elif onlymukk:
            # Function to be integrated to get mu2k
            def integrand(x):
                return ((1-abs(x))*indicator((-1 <= x <= 1))) * x ** kernelorder
            # Integrate to get mu2k
            import numpy as np
            from scipy.integrate import quad
            mukk = quad(integrand, -np.inf, np.inf)[0]
            return mukk
        else:
            return (1-abs(u))*int(-1 <= u <= 1)
    else:
        raise Exception('For triangular kernel only kernelorder=2 is available\n')


def epanechnikov(u, kernelorder, onlyroughness=False, onlymukk=False):
    """
    Epanechnikov kernel function
    :param u: input to kernel function
    :param kernelorder: order of the kernel function
    :param onlyroughness: if True only the roughness of the kernel is returned, a scalar defined as
    integrate_{-infty}^{intfty} (K(u))^2 du
    :param onlymukk: if True only  mukk := integrate_{-infty}^{intfty} u^kernelorder * K(u) du is returned
    :return: either K(u) or the roughness or mukk
    """

    if kernelorder == 2:
        if onlyroughness:
            # Function to be integrated to get roughness
            def integrand(x):
                return (0.75*(1 - x ** 2) * indicator((-1 <= x <= 1))) ** 2
            # Integrate to get roughness
            from scipy.integrate import quad
            r = quad(integrand, -np.inf, np.inf)[0]
            return r
        elif onlymukk:
            # Function to be integrated to get mu2k
            def integrand(x):
                return (0.75*(1 - x ** 2) * indicator((-1 <= x <= 1))) * x ** kernelorder
            # Integrate to get mu2k
            from scipy.integrate import quad
            mukk = quad(integrand, -np.inf, np.inf)[0]
            return mukk
        else:
            k = 0.75 * (1 - u ** 2) * indicator((-1 <= u <= 1))
            return k
    elif kernelorder == 4:
        if onlyroughness:
            # Function to be integrated to get roughness
            def integrand(x):
                kx = 0.75 * (1 - x ** 2) * indicator((-1 <= x <= 1))
                return (15 / 8 * (1 - 7 / 3 * x ** 2) * kx) ** 2
            # Integrate to get roughness
            from scipy.integrate import quad
            r = quad(integrand, -np.inf, np.inf)[0]
            return r
        elif onlymukk:
            # Function to be integrated to get mu2k
            def integrand(x):
                kx = 0.75 * (1 - x ** 2) * indicator((-1 <= x <= 1))
                return (15 / 8 * (1 - 7 / 3 * x ** 2) * kx) * x ** kernelorder
            # Integrate to get mu2k
            from scipy.integrate import quad
            mukk = quad(integrand, -np.inf, np.inf)[0]
            return mukk
        else:
            k = 0.75 * (1 - u ** 2) * indicator((-1 <= u <= 1))
            return 15 / 8 * (1 - 7 / 3 * u ** 2) * k
    elif kernelorder == 6:
        if onlyroughness:
            # Function to be integrated to get roughness
            def integrand(x):
                kx = 0.75 * (1 - x ** 2) * indicator((-1 <= x <= 1))
                return (175 / 64 * (1 - 6 * x ** 2 + 33 / 5 * x ** 4) * kx) ** 2
            # Integrate to get roughness
            from scipy.integrate import quad
            r = quad(integrand, -np.inf, np.inf)[0]
            return r
        elif onlymukk:
            # Function to be integrated to get mu2k
            def integrand(x):
                kx = 0.75 * (1 - x ** 2) * indicator((-1 <= x <= 1))
                return (175 / 64 * (1 - 6 * x ** 2 + 33 / 5 * x ** 4) * kx) * x ** kernelorder
            # Integrate to get mu2k
            from scipy.integrate import quad
            mukk = quad(integrand, -np.inf, np.inf)[0]
            return mukk
        else:
            k = 0.75 * (1 - u ** 2) * indicator((-1 <= u <= 1))
            return 175 / 64 * (1 - 6 * u ** 2 + 33 / 5 * u ** 4) * k
    else:
        raise Exception('For epanechnikov kernel only kernelorder in {2,4,6} is available')


def gaussian(u, kernelorder, onlyroughness=False, onlymukk=False, nthderivative=None):
    """
    Gaussian kernel function
    :param u: input to kernel function
    :param kernelorder: order of the kernel function
    :param onlyroughness: if True only the roughness of the kernel is returned, a scalar defined as
    integrate_{-infty}^{intfty} (K(u))^2 du
    :param onlymukk: if True only  mukk := integrate_{-infty}^{intfty} u^kernelorder * K(u) du is returned
    :param nthderivative: if True only the nth deivative of K(.) is retuned evaluated at u
    :return: either K(u) or the roughness or mukk or K^(n)(u)
    """

    if kernelorder == 2:
        if onlyroughness:
            # Function to be integrated to get roughness
            def integrand(x):
                return ((2 * np.pi) ** (-0.5) * np.exp(-0.5 * x ** 2)) ** 2
            # Integrate to get roughness
            from scipy.integrate import quad
            r = quad(integrand, -np.inf, np.inf)[0]
            return r
        elif onlymukk:
            # Function to be integrated to get mu2k
            def integrand(x):
                return ((2 * np.pi) ** (-0.5) * np.exp(-0.5 * x ** 2)) * x ** kernelorder
            # Integrate to get mu2k
            from scipy.integrate import quad
            mukk = quad(integrand, -np.inf, np.inf)[0]
            return mukk
        elif nthderivative == 2:
            k = (2 * np.pi) ** (-0.5) * np.exp(-0.5 * u ** 2)
            return -(k  - u ** 2 * k)
        else:
            k = (2 * np.pi) ** (-0.5) * np.exp(-0.5 * u ** 2)
            return k
    elif kernelorder == 4:
        if onlyroughness:
            # Function to be integrated to get roughness
            def integrand(x):
                kx = (2 * np.pi) ** (-0.5) * np.exp(-0.5 * x ** 2)
                return (0.5 * (3 - x ** 2) * kx) ** 2
            # Integrate to get roughness
            from scipy.integrate import quad
            r = quad(integrand, -np.inf, np.inf)[0]
            return r
        elif onlymukk:
            # Function to be integrated to get mu2k
            def integrand(x):
                kx = (2 * np.pi) ** (-0.5) * np.exp(-0.5 * x ** 2)
                return (0.5 * (3 - x ** 2) * kx) * x ** kernelorder
            # Integrate to get mu2k
            from scipy.integrate import quad
            mukk = quad(integrand, -np.inf, np.inf)[0]
            print('mu2k=', mukk)
            return mukk
        else:
            k = (2 * np.pi) ** (-0.5) * np.exp(-0.5 * u ** 2)
            return 0.5 * (3 - u ** 2) * k
    elif kernelorder == 6:
        if onlyroughness:
            # Function to be integrated to get roughness
            def integrand(x):
                kx = (2 * np.pi) ** (-0.5) * np.exp(-0.5 * x ** 2)
                return (1 / 8 * (15 - 10 * x ** 2 + x ** 4) * kx) ** 2
            # Integrate to get roughness
            from scipy.integrate import quad
            r = quad(integrand, -np.inf, np.inf)[0]
            return r
        elif onlymukk:
            # Function to be integrated to get mu2k
            def integrand(x):
                kx = (2 * np.pi) ** (-0.5) * np.exp(-0.5 * x ** 2)
                return (1 / 8 * (15 - 10 * x ** 2 + x ** 4) * kx) * x ** kernelorder
            # Integrate to get mu2k
            from scipy.integrate import quad
            mukk = quad(integrand, -np.inf, np.inf)[0]
            return mukk
        else:
            k = (2 * np.pi) ** (-0.5) * np.exp(-0.5 * u ** 2)
            return 1 / 8 * (15 - 10 * u ** 2 + u ** 4) * k
    else:
        raise Exception('For gaussian kernel only kerneloder in {2,4,6} is available')
