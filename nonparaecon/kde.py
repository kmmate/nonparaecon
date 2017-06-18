def kde_pdf(x, sampledata, kerneltype, bandwidth=None, bandwidthscale=None, kernelorder=2, biascorrected=True,
            bandwidth_bc=None, correctboundarybias=False, flowbound=None, fupbound=None, confidint=False,
            getsilverman=False, leaveoneout=False, leftoutindex=None):
    """
    Kernel density estimation for a given query point x0: fhat(x0). For both uni- and multivariate densities.
    :param x: scalar or d-vector (np.array for bias boundary bias correction): fhat(x) is returned
    :param sampledata: data used for estimation of dimension n*1 for scalar x or n*d for d-vector x (np.array)
    :param kerneltype: type of the kernel in kernels.py as string. For d-vector x product kernel is used
    :param kernelorder: order of the kernel function as integer, default is 2
    :param bandwidth: width of the kernel, scalar for scalar x; for d-vector x it can be a d-vector or scalar which is
    used for all d dimension. If not given, Silverman's rule of thumb is used for scalar (recommended).
    It has to be given for multivariate x and kernelorder of 4 and 6.
    :param bandwidthscale: if bandwidth is not given, Silverman's rule of thumb is used: bandwidthscale if given
    scales Silverman's bandwidth as bandwidth=(bandwidthscale)*(Silverman's bandwidth)
    :param biascorrected: use bias correction in estimation for asymptotic normality. Only for kernels of order 2,
    univariate x.  Recommended with Silvermans bandwidth
    :param bandwidth_bc: bandwidth used to estimate the bias correction term
    :param correctboundarybias: corrects for the boundary bias. If supp(X)=[a,b] then for every {x: (b-x)/h<1} and
     {x: (a-x)/h>-1}, where h is the bandwidth, the estimated kernel is evaluated at x:=b-h and x:=a+h respectively.
     Works for multivariate x as well with componentwise correction. flowbound or fupbound or both have to be given.
    :param flowbound: lower bound of the support of X. For multivariate, d-vector: [X1low,...,Xdlow] np.array
    :param fupbound: upper bound of the support of X For multivariate, d-vector: [X1up,...,Xdup] np.array
    :param confidint: boolean return confidence interval or not:
    if True the function returns the tuple (fhat, CI_low, CI_high) given as fhat+-2(fhat*R(k)/(n*bandwidth))^0.5, where
    R(k) is the roughness of the kernel. Only for scalar x.
    :param getsilverman: if True the only output is Silverman's bandwidth
    :param leaveoneout: if True mhat(x) is estimated in a leave-one-out fashion, leaving out the element
    indexed by leftoutindex. Note that division is done by n-1, not n
    :param leftoutindex: the index belonging to the left-out element when leaveoneout=True
    :return: estimated (boundary bias corrected) probability density function evaluated at the query point, fhat(x)
     or for scalar x: if 'confidint' is True a tuple (fhat(x), CI_low(x), CI_high(x)); only Silverman's optimal
     bandwidth if getsilverman=True
    """

    # Import kernels.py and numpy
    import numpy as np
    from scipy.integrate import quad
    import warnings
    from nonparaecon import kernels
    # Import the chosen kernel from kernels.py
    kernel = getattr(kernels, kerneltype)

    # Leave-one-out
    if leaveoneout:
        # error if True but no index is specified
        if leftoutindex is None:
            raise Exception('For leave-one-out version, leftoutindex must be specified')
        # drop observation otherwise
        else:
            sampledata = np.delete(sampledata, leftoutindex, axis=0)

    # Get sizes
    n = len(sampledata)
    d = np.size(x)

    # Issue warning if x is d-vector and confidint=True, also set it false and continue
    if (d>1 and confidint):
        warnings.warn('Confidence intervals are not available for multivariate distributions.'
                      'Program continues without producing confidence intervals.')


    # (I): Bandwidth
    # If no bandwidth is specified, use Silverman's rule of thumb for univariate x and kernelorder=2,
    #  raise error for multivariate x and kernelorder=4,6
    # Multivariate error
    if (d>1 and bandwidth is None):
        raise Exception('For multivariate density estimation bandwidth has to be given.')
    # kernelorder error
    elif ((bandwidth is None) and (kernelorder != 2)):
        raise Exception('For kernelorder other than 2, bandwidth has to be given. Silverman\'s can only be used with'
                        'kernelorder=2.')
    elif bandwidth is None:
        # Sigma^2 estimate
        sigma2hat = sampledata.var(0)
        # Compute || f''(x) ||_2^2 as a function of sigma2hat
        def integrand(y):
            f = 1 / (2 * np.pi * sigma2hat) ** 0.5 * np.exp(- y ** 2 / (2* sigma2hat))
            return (-1 / sigma2hat * (f - (y / sigma2hat) ** 2 * f)) ** 2
        f2primenormhat = quad(integrand, -np.inf, np.inf)[0]
        r = kernel(u=None, kernelorder=kernelorder, onlyroughness=True)
        mu2k = kernel(u=None, kernelorder=kernelorder, onlymukk=True)
        scale = (r / (mu2k ** 2 * f2primenormhat)) ** (1/5)
        # Silverman's optimal bandwidth
        # scale if bandwidthscale is given
        if not(bandwidthscale is None):
            bandwidth = bandwidthscale * scale * n ** (-1/5)
        else:
            bandwidth = scale * n ** (-1 / 5)
            if getsilverman:
                return bandwidth


    # (II): Boundary bias correction
    # Define the input array to the kernel adjusted for boundary bias for both uni- and multivariate x
    # If required correct for boundary bias
    # Only available for bounded support kernels, raise error for Gaussian
    if (correctboundarybias and kerneltype=='gaussian'):
        raise Exception('Boundary bias correction only works with bounded-support kernels.')
    # Raise error if boundary bias correction is required but no boundaries are given
    elif ((correctboundarybias and (flowbound is None)) and (correctboundarybias and (fupbound is None))):
        raise Exception('For boundary bias correction flowbound or fupbound or both has to be given.')
    # Do the adjustment, creates copy of original x
    elif correctboundarybias:
        # Copy the orginal for saving
        # Scalar
        if np.isscalar(x):
            x_c = x
        # Array
        else:
            x_c = x.copy()
        # For scalar
        if np.isscalar(x):
            if (not(flowbound is None) and ((flowbound - x) / bandwidth > -1)):
                x = flowbound + bandwidth
            if (not(fupbound is None) and ((fupbound - x) / bandwidth < 1)):
                x = fupbound - bandwidth
        # For multivariate
        else:
            if not(flowbound is None):
                x[(flowbound - x) / bandwidth > -1] = flowbound[(flowbound - x) / bandwidth > -1] \
                                                      + bandwidth[(flowbound - x) / bandwidth > -1]
            if not(fupbound is None):
                x[(fupbound - x) / bandwidth < 1] = fupbound[(fupbound - x) / bandwidth < 1] \
                                                    - bandwidth[(fupbound - x) / bandwidth < 1]
        u = (sampledata - x) / bandwidth
    # If no boundary bias adjustment is required go with the original values
    else:
        u = (sampledata - x) / bandwidth

    # (III): Estimation
    # (III.1): Univariate, scalar x case:
    if np.size(x) == 1:
        # Predict the density at the x
        fhat = 1 / (n * bandwidth) * sum([kernel(x, kernelorder) for x in u])
        # If required, estimate the bias correction term and construct the bias corrected estimator,
        # only for kernels of order 2
        # Error for kernelorder other than 2
        if (biascorrected and kernelorder != 2):
            raise Exception('Bias corrected estimator (biasccorrected=True) requires kernelorder=2')
        # Construct the estimator
        elif biascorrected:
            # Define a function which estimates the bias correction term

            def bc_term(x, sampledata, bandwidth, mu2k, bandwidth_bc=None):
                """
                Estimates the bias correction term for kernels of order 2. Uses Gaussian kernel
                with bandwidth bandwidth_bc to estimate f''(x)
                :param x: point at which the bias correction term is estimated
                :param sampledata: used to estimate f''(x)
                :param bandwidth: bandwidth of the kernel density estimator
                :param mu2k: mu2k of the kernel density estimator
                :param bandwidth_bc: bandwidth of the f''(x) estimator (2nd order Guassian kernel)
                :return: mu2k*(f''(x)hat)*bandwidth^2*0.5 where
                mu2k = integrate_{-infty}^{intfty} u^2 * K(u) du
                """
                # If bandwith_bc is not given set value such that bandwidth/bandwidth_bc --> epsilon>0 small
                if bandwidth_bc is None:
                    bandwidth_bc = bandwidth / (1 - np.finfo(float).eps)

                # Estimate f''(x)
                # Input to kernel
                u = (sampledata - x) / bandwidth_bc
                # Estimation
                f2primehat = 1 / (len(sampledata) * bandwidth_bc ** 3) *\
                    sum([kernels.gaussian(i, kernelorder=2, nthderivative=2) for i in u])

                # Return the bias correction terms
                return mu2k * f2primehat * bandwidth ** 2 * 0.5

            # Compute the correction term
            if correctboundarybias:
                correction = bc_term(x=x_c, sampledata=sampledata, bandwidth=bandwidth,
                                 mu2k=kernel(u=None, kernelorder=kernelorder, onlymukk=True),
                                 bandwidth_bc=bandwidth_bc)
            else:
                correction = bc_term(x=x, sampledata=sampledata, bandwidth=bandwidth,
                                 mu2k=kernel(u=None, kernelorder=kernelorder, onlymukk=True),
                                 bandwidth_bc=bandwidth_bc)
            # Compute the bias corrected estimator
            fbchat = fhat - correction
            # Raise warning if negative
            if fbchat < 0:
                warnings.warn('Estimated bias corrected density estimator is negative; problem for CI too.')
            fhat = fbchat
        # If required, compute and return the CI
        if not confidint:
            return fhat
        else:
            r = kernel(u=None, kernelorder=kernelorder, onlyroughness=True)
            ci_low = fhat - 1.96 * (fhat * r / (n * bandwidth)) ** 0.5
            ci_high = fhat + 1.96 * (fhat * r / (n * bandwidth)) ** 0.5
            return (fhat, ci_low, ci_high)
    # (III.2) Multivariate, d-vector x case
    else:
        if np.isscalar(bandwidth):
            fhat = 1 / (n * bandwidth ** d) * sum([np.prod([kernel(x, kernelorder) for x in u[i,:]])
                                                   for i in range(0,n)])
        else:
            fhat = 1 / (n * np.prod(bandwidth)) * sum([np.prod([kernel(x, kernelorder) for x in u[i, :]])
                                                       for i in range(0, n)])
        return fhat


def kde_cdf(x, sampledata, kerneltype, bandwidth=None, kernelorder=2):
    """
    Estimated cumulative distribution function for univariate distribution.
    :param x: Fhat(x) is returned
    :param sampledata: data used to estimate the density
    :param kerneltype: type of kernels.py to use as string
    :param bandwidth: bandwidth of the kernel. If None, Silverman's optimal bandwidth is used
    :param kernelorder: order of the kernel
    :return: Fhat(x)
    """
    # Import dependencies
    import numpy as np
    from scipy.integrate import quad

    # Import kernels.py
    from nonparaecon import kernels
    # Import the chosen kernel function from kernels.py
    kernel = getattr(kernels, kerneltype)

    # Get sizes
    n = len(sampledata)

    # If bandwidth is None, use Silverman's for kernelorder=2, otherwise raise error
    # kernelorder error
    if ((bandwidth is None) and (kernelorder != 2)):
        raise Exception('For kernelorder other than 2, bandwidth has to be given. Silverman''s can only be used with'
                        'kernelorder=2.')
    elif bandwidth is None:
        # Sigma^2 estimate
        sigma2hat = sampledata.var(0)

        # Compute || f''(x) ||_2^2 as a function of sigma2hat
        def integrand(y):
            f = 1 / (2 * np.pi * sigma2hat) ** 0.5 * np.exp(- y ** 2 / (2 * sigma2hat))
            return (-1 / sigma2hat * (f - (y / sigma2hat) ** 2 * f)) ** 2

        f2primehat = quad(integrand, -np.inf, np.inf)[0]
        r = kernel(u=None, kernelorder=kernelorder, onlyroughness=True)
        mu2k = kernel(u=None, kernelorder=kernelorder, onlymukk=True)
        scale = (r / (mu2k ** 2 * f2primehat)) ** (1 / 5)
        bandwidth = scale * n ** (-1 / 5)
        print('\nkde_cdf: Silverman''s bandwidth is used: ', bandwidth, '\n')

    # The upper limits of integration
    b = (sampledata - x) / bandwidth

    # The kernel function of the right order to integrate
    def tointegrate(u):
        return kernel(u, kernelorder=kernelorder)

    # Integrate
    Fhat = 1 / len(sampledata) * sum([quad(tointegrate, t, np.inf)[0] for t in b])
    return Fhat


def kde_quantile(alpha, sampledata, kerneltype, bandwidth=None, kernelorder=2):
    """
    Returns the requested percentile of the estimated kernel density
    :param alpha: scalar in [0,1] such that Phat(X<percentile_alpha)=alpha
    :param sampledata: data used to estimate the density
    :param kerneltype: type of kernel in kernels.py to use as string
    :param bandwidth: bandwidth of the kernel. If None, Silverman's optimal bandwidth is used
    :param kernelorder: order of the kernel
    :return: requested percentile, percentile_alpha such that P(X<percentile_alpha)=alpha
    """
    # Import dependencies
    import numpy as np
    from scipy.integrate import quad
    from scipy.optimize import fsolve

    # Import kernels.py
    from nonparaecon import kernels
    # Import the chosen kernel from kernels.py
    kernel = getattr(kernels, kerneltype)

    # Get sizes
    n = len(sampledata)

    # If bandwidth is None, use Silverman's for kernelorder=2, otherwise raise error
    # kernelorder error
    if ((bandwidth is None) and (kernelorder != 2)):
        raise Exception('For kernelorder other than 2, bandwidth has to be given. Silverman''s can only be used with'
                        'kernelorder=2.')
    elif bandwidth is None:
        # Sigma^2 estimate
        sigma2hat = sampledata.var(0)

        # Compute || f''(x) ||_2^2 as a function of sigma2hat
        def integrand(y):
            f = 1 / (2 * np.pi * sigma2hat) ** 0.5 * np.exp(- y ** 2 / (2 * sigma2hat))
            return (-1 / sigma2hat * (f - (y / sigma2hat) ** 2 * f)) ** 2

        f2primehat = quad(integrand, -np.inf, np.inf)[0]
        r = kernel(u=None, kernelorder=kernelorder, onlyroughness=True)
        mu2k = kernel(u=None, kernelorder=kernelorder, onlymukk=True)
        scale = (r / (mu2k ** 2 * f2primehat)) ** (1 / 5)
        bandwidth = scale * n ** (-1 / 5)
        print('\nkde_quantile: Silverman''s bandwidth is used: ', bandwidth, '\n')

    # The kernel function of the right order to integrate

    def tointegrate(u):
        return kernel(u, kernelorder=kernelorder)

    # Define the function to be optimised

    def tosolve(quantile):
        return (
            sum([quad(tointegrate, np.inf, (x - quantile) / bandwidth)[0] for x in sampledata])
            + len(sampledata) * alpha)

    # Solve the equation and return the requested percentile
    quantile = fsolve(tosolve, np.percentile(sampledata, q=100*alpha))[0]
    return quantile


def kde_plot(fhat, ismultiple, fsupport, plottitle, xlabel, ylabel, zlabel=None, color=None, alpha=None, fontsize=None,
             confidinton=False, confidint=None, ftrueon=False, ftrue=None,
             legendon=True, legendlabel=None, savemode=False, filepath=None, viewmode=True):
    """
   Creates plot to visualise kernel density estimation (multiple multivariate is not available yet: the actual plotting
   implementation is missing, until then it's worked out)
   :param fhat: predicted density fhat(x0). For a single univariate distribution it is an array. For multiple univariate
   distributions and a single multivariate distribution it is an array of arrays.
   For multiple multivariate distributions it a list of arrays of arrays.
   For multivarate at most J=2 distributions are allow allowed
   :param ismultiple: boolean, indicates whether fhat contains multiple distributions
   :param fsupport: Univariate f: support of f present in the plot; for multiple univariate distributions the same
   support is used. Bivariate f: list such that [rangeX1, rangeX2].
   :param plottitle: title of the plot
   :param xlabel: label on the x-axis
   :param ylabel: label on the y-axis
   :param zlabel: label on the z-axis, only for bivariate
   :param color: dictionary {'fhat_j': color of fhat_j, 'confidintfill_j': color to fill confidint_j,
   'ftrue_j': color of ftrue_j} where j=1,...J is the number of distribution if fhat is multiple;
   For J=1, 'j' subscripts should not be used. confidint and ftrue are optional can be specified if
   function called with confidint and ftrue. Bivariate f: only 'fhat_j' and 'ftrue_j' are used, both should be colormap
   and j=1,2 if multiple, 'j' subscripts are to be drpped if not multiple
   :param alpha: dictionary {'fhat_j': alpha of fhat_j, 'confidintfill_j': alpha to fill confidint_j,
   'ftrue_j': alpha of ftrue_j} where j=1,...J is the number of distribution if fhat is multiple;
   For J=1, 'j' subscripts should not be used. confidint_j and ftrue_j are optional  can be specified if function is
    called with confidint and ftrue. Only for univariate f
   :param fontsize: dictionary {'title': (), 'xlabel': (), 'ylabel':(), 'xticklabel':(), 'yticklabel':(), 'legend':()}
   with the corresponding sizes in ().
   :param confidinton: scalar boolean or list of booleans: (i) scalar: True or False option applies to all
   j=1,...,J distributions (ii) list: if 'j'th element is True confidence intervals showed for the
   'j'th distribution; length of list is J; requires confidint_j to be given. Only for univariate f
   :param confidint: dictionary, confidendence intervals {'low_j': ci_low_j, 'high_j': ci_high_j} for j=1,...J if
   fhat is multiple. For J=1, 'j' subscripts should not be used. Only for univariate f
   :param ftrueon: scalar boolean of list of booleans (i) scalar: if True the true density is shown
   (ii) list of booleans: if the 'j'th element is true ftrue_j is shown for the 'j'th distribution; length of list is J;
   requires ftrue_j to be given
   :param ftrue: array of array of arrays: (i) array: for J=1 single distribution the true density
   (ii) array of arrays: array of length J, 'j'th array corresponds to the true distribution of the 'j'th distribution
   :param legendon: if True legend is shown. Only for univariate f
   :param legendlabel: dictionary: (i) single distribution J=1:
   {'fhat': label of fhat in the legend, 'ftrue': label of ftrue in the legend,
   'confidint': label of confidint in the legend}
   (ii) for multiple distributions J>1: {'dist_j': label of the 'j'th distribution}, no separate labels for ftrue or
   confidint. Only for univariate f
   :param savemode: if True figure is saved in the current working directory, requires filename to be given
   :param filepath: name under which  figure is saved as r'destination\filename' without file extension
   :param viewmode: if True the figure is displayed, blocking the execution
   :return: shows the figure if required so; save the figure (to a png and latex pgf file)
   """
    # Import dependencies set formatting
    import numpy as np
    # Setting plotting formats
    import matplotlib as mpl
    if savemode:
        # If filename missing throw error
        if filepath is None:
            filepath = input('For savemode=True, filepath has to be specified: ')

        def figsize(scale):
            fig_width_pt = 469.755  # Get this from LaTeX using \the\textwidth
            inches_per_pt = 1.0 / 72.27  # Convert pt to inch
            golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
            fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
            fig_height = fig_width * golden_mean  # height in inches
            fig_size = [fig_width, fig_height]
            return fig_size

        pgf_with_rc_fonts = {'font.family': 'serif', 'figure.figsize': figsize(0.9), 'pgf.texsystem': 'pdflatex'}
        mpl.rcParams.update(pgf_with_rc_fonts)
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['text.latex.unicode'] = True
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    plt.rc('font', family='serif')

    # Clear previous figure
    plt.close()
    # Check if bivariate
    is_bivariate = isinstance(fsupport, list)
    # Get the number of distributions if J>1
    if ismultiple:
        J = len(fhat)

    # Set dafault values of fontsize, color, alpha, legendlabel
    # fontsize
    if fontsize is None:
        fontsize = dict({'title': 12, 'xlabel': 11, 'ylabel': 11, 'zlabel': 11,
                         'xticklabel': 10, 'yticklabel': 10, 'zticklabel': 10,
                         'legend': 10})
    # color, alpha, legendlabel
    # J=1 single distribution
    if not ismultiple:
        # Univariate
        if not is_bivariate:
            if color is None:
                color = dict({'fhat': 'b', 'confidintfill': 'royalblue', 'ftrue': 'red'})
            if alpha is None:
                alpha = dict({'fhat': 1, 'confidintfill': 0.6, 'ftrue': 1})
            if legendlabel is None:
                legendlabel = dict({'fhat': 'Estimated', 'ftrue': 'True', 'confidint': 'CI'})
        # Multivariate
        else:
            if color is None:
                color = dict({'fhat': 'plasma', 'ftrue': 'Greys'})
    # J>1 multiple distributions
    else:
        vmi = 1
        # Univariate
        if not is_bivariate:
            # Color
            # List of colors for the distributions. Use the same color with different alphas and linestyle
            # to visualise confidint and ftrue
            if color is None:
                color = dict()
                for j in range(1, J + 1):
                    actualcolor = tuple(np.random.rand(3))
                    color['fhat_' + str(j)] = actualcolor
                    color['confidintfill_' + str(j)] = actualcolor
                    color['ftrue_' + str(j)] = actualcolor
            # Alpha
            if alpha is None:
                alpha = dict()
                for j in range(1, J + 1):
                    alpha['fhat_' + str(j)] = 1
                    alpha['confidintfill_' + str(j)] = 0.6
                    alpha['ftrue_' + str(j)] = 1
            # Legend
            if legendlabel is None:
                legendlabel = dict()
                for j in range(1, J + 1):
                        legendlabel['dist_' + str(j)] = 'Dist. ' + str(j)
        # Multivariate
        else:
            if color is None:
                color = dict({'fhat_1': 'plasma', 'ftrue_1': 'Greys', 'fhat_2': 'cool', 'ftrue_2': 'Purples'})

    # Make the basic plot
    # Univariate
    if not is_bivariate:
        f, ax = plt.subplots()
        # Add true density if required
        # J=1 single distribution
        if not ismultiple:
            if ftrueon:
                if ftrue is None:
                    raise Exception('If ftrueon=True ftrue is required')
                else:
                    ax.plot(fsupport, ftrue, label=legendlabel['ftrue'], color=color['ftrue'], alpha=alpha['ftrue'])
            # Estimated density
            ax.plot(fsupport, fhat, label=legendlabel['fhat'], color=color['fhat'], alpha=alpha['fhat'])
            # Add confidence intervals if required
            if confidinton:
                if confidint is None:
                    raise Exception('If confidinton=True, confidint is required.')
                else:
                    ax.fill_between(fsupport, confidint['ci_low'], confidint['ci_high'],
                                    label=legendlabel['confidint'],
                                    facecolor=color['confidintfill'], alpha=alpha['confidintfill'])
        # J>1 multiple distributions
        else:
            # Cheking and setting ftrueon and confidinton
            # ftrueon
            # check if one boolean or list, set to J-long list if one boolean
            # one boolean
            if isinstance(ftrueon, bool):
                # error if ftruon is True but no ftrue is given
                if ftrueon and ftrue is None:
                    raise Exception('For ftrueon=True, ftrue must be given.\n')
                # error if one boolean ftrueon=True is given but ftrue is only one array
                if ftrueon and np.ndim(ftrue) != J:
                    raise Exception(['If ftruenon=True scalar boolean, ftrue must be an array of arrays of length J' +
                                    ' with corresponding arrays belonging to the ''j''th distribution'][0])
                # list
                ftrueon_list = [ftrueon] * J
            # list of boolean
            else:
                # error if the number of Trues in ftrueon list is not the same as the number of arrays in ftrue
                num_ftrueon = sum([int(b) for b in ftrueon])
                if num_ftrueon != len(ftrue):
                    raise Exception('The number of Trues in ftrueon doesn''t match the number of arrays in ftrue.\n')
                ftrueon_list = ftrueon
            # confidinton
            # # check if one boolean or list, set to J-long list if one boolean
            # one boolean
            if isinstance(confidinton, bool):
                # error if True but no confidint is given
                if confidinton and confidint is None:
                    raise Exception('For confidinton=True, confidint must be given.\n')
                # list
                confidinton_list = [confidinton] * J
                # list of boolean
            else:
                # error if the number of Trues in ftrueon list is not the same as the number of arrays in ftrue
                num_confidinton = sum([int(b) for b in confidinton])
                if num_confidinton != len(confidint) / 2:
                    raise Exception(['The number of Trues in confidinton doesn''t match the number of ci_low, ci_high' +
                     ' pairs in confidint.\n'][0])
                confidinton_list = confidinton

            # Plotting
            for j in range(1, J+1):
                if ftrueon_list[j-1]:
                    ax.plot(fsupport, ftrue[j-1], linestyle='--',
                            color=color['ftrue_' + str(j)], alpha=alpha['ftrue_' + str(j)])
                # Estimated density
                ax.plot(fsupport, fhat[j-1], label=legendlabel['dist_' + str(j)],
                        color=color['fhat_' + str(j)], alpha=alpha['fhat_' + str(j)])
                # Add confidence intervals if required
                if confidinton_list[j-1]:
                        ax.fill_between(fsupport, confidint['ci_low_' + str(j)], confidint['ci_high_' + str(j)],
                                        facecolor=color['confidintfill_' + str(j)],
                                        alpha=alpha['confidintfill_' + str(j)])
        # Add legend if required
        if legendon:
            ax.legend(fontsize=fontsize['legend'])
    # Bivariate
    else:
        x1, x2 = np.meshgrid(fsupport[0], fsupport[1])
        f = plt.figure()
        ax = f.gca(projection='3d')
        # Add true density if required
        if ftrueon:
            if ftrue is None:
                raise Exception('If ftrueon=True ftrue is required')
            else:
                ax.plot_surface(x1, x2, ftrue, cmap=color['ftrue'])
        # Estimated density
        ax.plot_surface(x1, x2, fhat, cmap=color['fhat'])
        plt.setp(ax.get_zticklabels(), size=fontsize['zticklabel'])
        ax.set_zlabel(zlabel, size=fontsize['zlabel'])
    ax.set_title(plottitle, size=fontsize['title'])
    plt.setp(ax.get_xticklabels(), size=fontsize['xticklabel'], ha='center')
    plt.setp(ax.get_yticklabels(), size=fontsize['yticklabel'])
    plt.xlabel(xlabel, size=fontsize['xlabel'])
    plt.ylabel(ylabel, size=fontsize['ylabel'])

    # Save figure if requires
    if savemode:
        f.set_size_inches(figsize(1.5)[0], figsize(0.9)[1])
        # save to pgf
        plt.savefig(filepath+'.pgf', bbox_inches='tight')
        # save to png
        plt.savefig(filepath+'.png', bbox_inches='tight')

    # Show figure if reguired
    if viewmode:
        plt.show(block=viewmode)
