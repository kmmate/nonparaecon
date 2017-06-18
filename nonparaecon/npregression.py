"""
Functions for nonparametic regression
"""

# Import dependencies
import numpy as np
import numba
from sklearn import preprocessing
from nonparaecon import kernels
from _warnings import warn

# Define squared error loss
# defining squared loss

def l2loss(yhat, yactual):
    """
    Return the sum of squared errors base on the true and estimated data vector/list/array
    :param yhat: estimated values
    :param yactual: true values
    :return: sum of squared errors
    """
    errorhat = np.array(yhat) - np.array(yactual)
    sse = (errorhat.T).dot(errorhat)
    return sse



def npregression_nadwats(x, xdata, ydata, kerneltype, kernelorder=2, bandwidth=None):
    """
    Nadaraya-Watson nonparametric regression estimator
    :param x: such that mhat(x) is computed; scalar or d-vector with no constant
    :param xdata: sample data for x; n*d array, n: number of observations, d: number of elements in x (no constant)
    :param ydata: sample data for y; n-array, n:number of oservations
    :param kerneltype: type of the kernel used for weighting; string, a name of a kernel in kernels.py; for multivariate
    x, product kernel is used
    :param kernelorder: order of the kernel used for weighting, integer in {2, 4, 6}
    :param bandwidth: bandwidth of the kernel; scalar or d array
    :return: point estimate mhat(x)
    """

    # Get sizes, dimensions
    dim = np.ndim(x)
    n = len(ydata)

    # Errors
    # error if sampledata are not of the same length
    if len(ydata) != len(xdata):
        raise Exception('Length of ydata and xdata must be the same')
    # error if the dimension of x differs from xdata
    if np.isscalar(x):
        if np.ndim(xdata) != 1:
            raise Exception('Dimension of x and xdata must be the same')
    elif len(x) != np.size(xdata, 1):
        raise Exception('Dimension of x and xdata must be the same')

    # Get kernel
    kernel = getattr(kernels, kerneltype)
    # input to kernel
    u = (xdata - x) / bandwidth

    # Scalar x case
    if dim == 0:
        denominator = sum([kernel(i, kernelorder) for i in u])
        numerator = sum([ydata[i] * kernel(u[i], kernelorder) for i in range(0, n)])
        mhat = numerator / denominator
        return mhat
    else:
        denominator = sum([np.prod([kernel(k, kernelorder) for k in i]) for i in u])
        numerator = sum([ydata[i] * np.prod([kernel(k, kernelorder) for k in u[i]]) for i in range(0, n)])
        mhat = numerator / denominator
        return mhat




def npregression_locpol(x, xdata, ydata, poldegree, kerneltype, kernelorder=2, bandwidth=None,
                        get_se=False, get_ci=False, leaveoneout=False, leftoutindex=None):
    """
    Local polynomial (linear) regression. For x with more than one elements, the maximum polynomial degree available is
    two. The returned confidence intervals are not bias corrected.
    :param x: such that mhat(x) is computed; scalar or d-vector with no constant
    :param xdata: sample data for x; n*d array, n: number of observations, d: number of elements in x (no constant)
    :param ydata: sample data for y; n-array, n:number of oservations
    :param poldegree: order of the polynomial; integer, at most 2 for d>=2
    :param kerneltype: type of the kernel used for weighting; string, a name of a kernel in kernels.py; for multivariate
    x, product kernel is used
    :param kernelorder: order of the kernel used for weighting, integer in {2, 4, 6}
    :param bandwidth: bandwidth of the kernel; scalar or d array
    :param get_se: if True, the standard errors are also returned; computed
    :param get_ci: if True confidence intervals are also returned
    :param leaveoneout: if True mhat(x) is estimated in a leave-one-out fashion, leaving out the element
    indexed by leftoutindex
    :param leftoutindex: the index belonging to the left-out element when leaveoneout=True
    :return: mhat or (mhat, se) or (mhat, CI_low, CI_high) or (mhat, se, CI_low, CI_high)
    """


    # Get sizes, dimensions
    dim = np.ndim(x)
    n = len(ydata)

    # Errors
    # error if sampledata are not of the same length
    if len(ydata) != len(xdata):
        raise Exception('Length of ydata and xdata must be the same')
    # error if the dimension of x differs from xdata
    if np.isscalar(x):
        if np.ndim(xdata) != 1:
            raise Exception('Dimension of x and xdata must be the same')
    elif isinstance(x, np.matrix):
        if np.ndim(x) != np.size(xdata, 1):
            raise Exception('Dimension of x and xdata must be the same')
    # error if leave-one-out mode is on but no index is specified
    if leaveoneout and leftoutindex is None:
        raise Exception('When leaveoneout=True, leftoutindex must be specified')

    # y to array
    ydata = np.array(ydata)
    # design matrix with desired polynomials and constant
    if np.isscalar(x):
        xdiff = xdata[:, None] - x
    else:
        xdiff = xdata - x
    X = preprocessing.PolynomialFeatures(degree=poldegree, include_bias=True).fit_transform(xdiff)

    # Kernel weights
    # get kernel
    kernel = getattr(kernels, kerneltype)
    # input to kernel
    u = (xdata - x) / bandwidth
    # simple kernel scalar x
    if dim == 0:
        w = np.array([kernel(i, kernelorder) ** 0.5 for i in u])
    # product kernel d-vector x
    else:
        w = np.array([np.product([kernel(np.array(k), kernelorder) for k in i]) ** 0.5 for i in u])
    # Transform variables, elementwise multiplication
    Xtilde = X * w[:, None]
    ytilde = ydata * w
    #print(w)


    # Leave-one-out mode, drop the specified observation from
    if leaveoneout:
        Xtilde = np.delete(Xtilde, leftoutindex, axis=0)
        ytilde = np.delete(ytilde, leftoutindex, axis=0)

    # Estimate beta
    XtildeT = Xtilde.T
    XtildeT_Xtildeinv = np.linalg.inv(XtildeT.dot(Xtilde))
    betehat_p1 = XtildeT_Xtildeinv.dot(XtildeT)
    betahat = betehat_p1.dot(ytilde)
    mhat = betahat[0]

    # Return mhat(x) if SE is not required
    if (not get_se) and (not get_ci):
        return mhat
    # Compute variance and return SE
    else:
        # uhat
        uhat = ytilde - Xtilde.dot(betahat)
        # varhat as parts
        try:
            s2hat = (uhat.T).dot(uhat) / (n - len(x) - 1)
        except:
            s2hat = (uhat.T).dot(uhat) / (n - 2)
        varbetahat = s2hat * XtildeT_Xtildeinv
        se_mhat = np.sqrt(varbetahat[0, 0])
        # if required return se
        if get_se and not get_ci:
            return mhat, se_mhat
        elif get_ci:
            ci_low = mhat - 1.96 * se_mhat
            ci_high = mhat + 1.96 * se_mhat
            ci = [ci_low, ci_high]
            if get_se:
                return mhat, se_mhat, ci
            else:
                return mhat, ci



def npregression_locpollscv(searchrange, xdata, ydata, poldegree, kerneltype, kernelorder=2, subsamplesize=None,
                            get_sse=False):
    """
    Least squares cross validation for local polynomial regression to find optimal bandwidth
    :param searchrange: range in which the optimal bandwidth is searched
    :param xdata: sample data for x; n*d array, n: number of observations, d: number of elements in x (no constant)
    :param ydata: sample data for y; n-array, n:number of oservations
    :param poldegree: order of the polynomial; integer, at most 2 for d>=2
    :param kerneltype: type of the kernel used for weighting; string, a name of a kernel in kernels.py; for multivariate
                        x, product kernel is used
    :param kernelorder: order of the kernel used for weighting, integer in {2, 4, 6}
    :param subsamplesize: size of th subsample used for leave-one-out cross validation. Defaults to the original number
                          of observations
    :param get_sse: if True, also the list of sum of squared errors (SSE) is returned corresponding
    to each bandwidth in searchrange
    :return: optimal bandwidth or (optimal bandwidth, list of SSE). If subsamplesize is given, the optimal bandwidth
    found in searchrange is adjusted for subsampling
    """


    # Get size, dimension
    n = len(ydata)
    try:
        d = np.size(xdata[0], 1)
    except:
        d = 1
    print('Perceived  number of variables in x: ', d)


    # Draw a subsample for validation
    if not(subsamplesize is None):
        # default random gen
        np.random.seed([0])
        # draw subsample
        permindex = np.random.permutation(len(ydata))[0:subsamplesize]
        y_sub = np.array(ydata)[permindex]
        X_sub = np.array(xdata)[permindex]
        n_s = len(y_sub)
    else:
        y_sub = np.array(ydata)
        X_sub = np.array(xdata)
        n_s = n

    # For each h in the range, estimate y_i with the leave-one-out loc. pol. reg. estimator evaluated at x_i
    sse_list = list()
    hround = 0
    for h in searchrange:
        if d == 1:
            yhat = [npregression_locpol(x=X_sub[i], xdata=X_sub, ydata=y_sub, kerneltype=kerneltype,
                                        kernelorder=kernelorder, poldegree=poldegree, bandwidth=h, get_ci=False,
                                        get_se=False, leaveoneout=True, leftoutindex=i) for i in range(n_s)]
        else:
            yhat = [npregression_locpol(x=X_sub[i], xdata=X_sub, ydata=y_sub, kerneltype=kerneltype,
                                        kernelorder=kernelorder, poldegree=poldegree, bandwidth=h, get_ci=False,
                                        get_se=False, leaveoneout=True, leftoutindex=i) for i in range(n_s)]
        sse = l2loss(yhat=yhat, yactual=y_sub)
        sse_list.append(sse)
        # best h
        # for the first round let h be the best h
        if hround == 0:
            best_h_sub = h
        elif (sse <= np.array(sse_list)).sum() == len(sse_list):
            best_h_sub = h
        # scale it back to original sample size (adjust for subsampling)
        best_h = best_h_sub * (n / n_s) ** (-1 / (2 * poldegree + d + 2))
        # step round
        hround = hround + 1

    # issue warning if best_h_sub is on the boundary of searchrange
    if best_h_sub == searchrange[0]:
        warn('The found optimal bandwidth is on the lower boundary of searchrange. Consider expanding '
             'searchrange to the left.')
    elif best_h_sub == searchrange[-1]:
        warn('The found optimal bandwidth is on the upper boundary of searchrange. Consider expanding '
             'searchrange to the right.')
    # return the optimal bandwidth and/or list of SSE and subsample size corrected bandwidth optionally
    if not get_sse:
        return best_h
    else:
        return best_h, sse_list



def npregression_rdd(design, ydata, runningxdata, cutoff, treatmentabove, bandwidth, poldegree, kerneltype='triangular',
                     kernelorder=2, get_se=False, sepoldegree=None, get_ci=False):
    """
    Nonparametric regression discontinuity design.

     It returns the estimate of conditional average treatment effect, tau := E[Y(1)-Y(0) | X=cutoff].

    PARAMETERS
    ----------
    :param design: 'sharp' or 'fuzzy' (as of now only 'sharp' is available)
    :param ydata: data on the dependent variable
    :param runningxdata: data on the running variable
    :param cutoff: cutoff determining the treatment, D
    :param treatmentabove: if True D=1(runningx > cutoff); if False D=1(runningx < cutoff)
    :param bandwidth: bandwidth to use both for point estimate and standard error estimate
    :param poldegree: degree of polynomial used for point-estimate
    :param kerneltype: kerneltype, integer from kernely.py
    :param kernelorder: order of the kernel used for weighting, integer in {2, 4, 6}
    :param get_se: if True also the standard error is returned
    :param sepoldegree: polynomial degree used to produce the standard error. By default sepoldegree=poldegree+1
                        to reduce bias
    :param get_ci: if True also the confidence interval is returned

    RETURNS
    -------
    :return: tauhat or (tauhat, sehat_tau) or (tauhat, ci) or (tauhat, sehat_tau, ci) with ci=[ci_low, ci_high]
    """

    # Sharp design
    if design == 'sharp':
        # treatment above or below cutoff
        if treatmentabove:
            d = runningxdata > cutoff
        else:
            d = runningxdata < cutoff
        # if se or ci is required, give default sepoldegree
        if (get_se or get_ci) and sepoldegree is None:
            sepoldegree = poldegree + 1
        # compute muhat's and also standard errors if the polynomial degree matches
        if (get_se or get_ci) and sepoldegree == poldegree:
            # treated group
            muhat_t, sehat_t = npregression_locpol(x=cutoff, xdata=runningxdata[d], ydata=ydata[d], poldegree=poldegree,
                                                   kerneltype=kerneltype, kernelorder=kernelorder, bandwidth=bandwidth,
                                                   get_se=True)
            # control group
            muhat_c, sehat_c = npregression_locpol(x=cutoff, xdata=runningxdata[d == False], ydata=ydata[d == False],
                                                   poldegree=poldegree, kerneltype=kerneltype, kernelorder=kernelorder,
                                                   bandwidth=bandwidth, get_se=True)
            # combined se
            sehat_tau = np.sqrt(sehat_t ** 2 + sehat_c ** 2)
        # if the polynomial order differ compute muhat, se and ci differently if requested
        # muhat
        # treated
        muhat_t = npregression_locpol(x=cutoff, xdata=runningxdata[d], ydata=ydata[d], poldegree=poldegree,
                                      kerneltype=kerneltype, kernelorder=kernelorder, bandwidth=bandwidth)
        # control
        muhat_c = npregression_locpol(x=cutoff, xdata=runningxdata[d == False], ydata=ydata[d == False],
                                      poldegree=poldegree, kerneltype=kerneltype, kernelorder=kernelorder,
                                      bandwidth=bandwidth)
        # sehat and ci if requested
        if get_se or get_ci:
            # treated group
            mhat, sehat_t = npregression_locpol(x=cutoff, xdata=runningxdata[d], ydata=ydata[d], poldegree=sepoldegree,
                                                   kerneltype=kerneltype, kernelorder=kernelorder, bandwidth=bandwidth,
                                                   get_se=True)
            # control group
            mhat, sehat_c = npregression_locpol(x=cutoff, xdata=runningxdata[d == False], ydata=ydata[d == False],
                                                poldegree=sepoldegree, kerneltype=kerneltype, kernelorder=kernelorder,
                                                bandwidth=bandwidth, get_se=True)
            # combined se
            sehat_tau = np.sqrt(sehat_t ** 2 + sehat_c ** 2)

        # estimated tau
        tauhat = muhat_t - muhat_c

        # ci if needed
        if get_ci:
            ci_low = tauhat - 1.96 * sehat_tau
            ci_high = tauhat + 1.96 * sehat_tau
            ci = [ci_low, ci_high]

        # returns
        if not get_se and not get_ci:
            return tauhat
        if get_se and not get_ci:
            return tauhat, sehat_tau
        if not get_se and get_ci:
            return tauhat, ci
        if get_se and get_ci:
            return tauhat, sehat_tau, ci



def npregression_rddlscv(design, searchrange, ydata, runningxdata, cutoff, treatmentabove, poldegree,
                         kerneltype='triangular', kernelorder=2, subsamplesize=None, get_sse=False):
    """
    Least squares cross validation for (sharp) regression discontinuity design.

    PARAMETERS
    ----------
    :param design: 'sharp' or 'fuzzy' (as of now onlya 'sharp' design is available)
    :param searchrange: range in which the bandwidth is searched
    :param ydata: data on the dependent variable
    :param cutoff: cutoff determining the treatment, D
    :param treatmentabove: if True D=1(runningx > cutoff); if False D=1(runningx < cutoff)
    :param poldegree: degree of the polynomial used to obtain the point estimate
    :param kerneltype: kerneltype, integer from kernely.py
    :param kernelorder: order of the kernel used for weighting, integer in {2, 4, 6}
    :param subsamplesize: size of th subsample used for leave-one-out cross validation. Defaults to the original number
                          of observations
    :param get_sse: if True also the sum of squared errors are returned as a list

    RETURNS
    -------
    :return: the optimal bandwidth or (optimal bandwidth, sse list) with optimal bandwidth adjusted for subsamplesize
             in both cases
    """


    # get size
    n = len(ydata)
    # do subsampling if requested
    if not(subsamplesize is None):
        # default random gen
        np.random.seed([0])
        # draw subsample
        permindex = np.random.permutation(len(ydata))[0:subsamplesize]
        y_sub = np.array(ydata)[permindex]
        runningx_sub = np.array(runningxdata)[permindex]
        n_sub = len(y_sub)
    else:
        y_sub = np.array(ydata)
        runningx_sub = np.array(runningxdata)
        n_sub = n


    # Sharp design
    if design == 'sharp':
        # treatment above or below cutoff
        if treatmentabove:
            d_sub = runningx_sub > cutoff
        else:
            d_sub = runningx_sub < cutoff

        # the data used for loss-computing
        # boolean mask
        lowlimit = np.median(runningx_sub[runningx_sub < cutoff])
        upperlimit = np.median(runningx_sub[runningx_sub > cutoff])
        mask1 = np.all([(lowlimit <= runningx_sub), (runningx_sub <= upperlimit)], axis=0)
        # apply mask
        y_loss = y_sub[mask1]
        d_loss = np.array(d_sub)[mask1]
        runningx_loss = runningx_sub[mask1]
        # get size
        n_loss = len(y_loss)

        # find optimal bandwidth
        # preallocation
        muhat_h = np.empty((n_loss, ))
        sse_list = list()
        hround = 0
        # looping through bandwidths
        for h in searchrange:
            # loop through observations
            for i in range(n_loss):
                # if i is below cutoff...
                if runningx_loss[i] < cutoff:
                    # use observations only below it from the 'whole' sample (subsample if required) to predict y
                    mask2 = runningx_sub < runningx_loss[i]
                    # predict
                # if i is above cutoff...
                else:
                    # use observations oly above it from the 'whole' sample (subsample if required) to predict y
                    mask2 = runningx_sub > runningx_loss[i]
                # predict y
                muhat_h[i] = npregression_locpol(x=runningx_loss[i], xdata=runningx_sub[mask2], ydata=y_sub[mask2],
                                                 poldegree=poldegree, kerneltype=kerneltype, kernelorder=kernelorder,
                                                 bandwidth=h)
            # calculate the sse belonging to the current h
            sse_h = l2loss(yhat=muhat_h, yactual=y_loss)
            sse_list.append(sse_h)
            # best h
            # for the first round let h be the best h
            if hround == 0:
                best_h_sub = h
            # than h can only be the best if the sse belonging to it is smaller than or equal to all previous ones
            elif (sse_h <= np.array(sse_list)).sum() == len(sse_list):
                best_h_sub = h
            # scale it back to original sample size (adjust for subsampling)
            d = 1
            best_h = best_h_sub * (n / n_sub) ** (-1 / (2 * poldegree + d + 2))
            # step round
            hround += 1

            # returns
            if not get_sse:
                return best_h
            else:
                return best_h, sse_list



def npregression_plot(mhat, ismultiple, xdomain, plottitle, xlabel, ylabel, color=None, alpha=None,
                      fontsize=None, confidinton=False, confidint=None, mtrueon=False, mtrue=None,
                      truestyle=None, legendon=True, legendlabel=None, subplottitle=None, seriesperrow=None,
                      savemode=False, filepath=None, viewmode=True):
    """
   Creates plot to visualise nonparametric regression (multiple (x,y) (x 2-vector) series is not available yet:
   the actual plotting implementation is missing, until then it's worked out)

   PARAMETERS
   ----------
   :param mhat: predicted density mhat(x0). For a single (x,y) series it is an array.
                For multiple (x,y) series it is a list of arrays.
                For 2-vector x at most J=2 series are allow allowed
   :param ismultiple: boolean, indicates whether mhat contains multiple (x,y) series, e.g. {(x,y1,y2,y2)} data set
   :param xdomain: scalar x: support of x present in the plot; for multiple (x,y) (x scalar) series the same
                   support is used: (x,y1,y2,..yJ).
                   2-vector x: list such that [rangeX1, rangeX2].
   :param plottitle: title of the plot
   :param xlabel: label on the x-axis
   :param ylabel: label on the y-axis
   :param color: Scalar x: dictionary {'mhat_j': color of mhat_j, 'confidintfill_j': color to fill confidint_j,
                 'mtrue_j': color of mtrue_j} where j=1,...J is the number of y's if mhat is multiple;
                  For J=1, 'j' subscripts should not be used. confidint and ftrue are optional can be specified if
                  function called with confidint and ftrue. 2-vector x: colormap
   :param alpha: dictionary {'mhat_j': alpha of mhat_j, 'confidintfill_j': alpha to fill confidint_j,
                 'mtrue_j': alpha of mtrue_j} where j=1,...J is the number of y's if mhat is multiple;
                 For J=1, 'j' subscripts should not be used. confidint_j and mtrue_j are optional  can be specified
                  if function is called with confidint and mtrue. Only for scalar x
   :param fontsize: dictionary {'title': (), 'xlabel': (), 'ylabel':(), 'xticklabel':(), 'yticklabel':(), 'legend':()}
                    with the corresponding sizes in ().
   :param confidinton: scalar boolean or list of booleans: (i) scalar: True or False option applies to all
                       j=1,...,J series; list: if 'j'th element is True confidence intervals showed for the
                       'j'th series; length of list is J; requires confidint_j to be given. Only for scalar x
   :param confidint: dictionary, confidendence intervals {'low_j': ci_low_j, 'high_j': ci_high_j} for j=1,...J if
                     mhat is multiple. For J=1, 'j' subscripts should not be used. Only for scalar x
   :param mtrueon: scalar boolean or list of booleans (i) scalar: if True the true y is shown
                    (ii) list of booleans: if the 'j'th element is true mtrue_j is shown for the 'j'th series;
                    length of list is J; requires mtrue_j to be given. For 2-vetor x it must be a scalar: mtrue is
                    either shown for all or none of the series
   :param mtrue: array or list of arrays: (i) array: for J=1 the true y
                 (ii) list of arrays: array of length J, 'j'th array corresponds to the true y of the 'j'th series.
   :param truestyle: style of mtrue, e.g. '*'. Only for scalar x
   :param legendon: if True legend is shown
   :param legendlabel: dictionary (i) single series J=1:
                       (i.1) scalar x: {'mhat': label of mhat in the legend, 'mtrue': label of mtrue in the legend,
                       'confidint': label of confidint in the legend}
                        (i.2) 2-vector x: {'cbar_trueoff': label of colorbar when mtrue is not shown,
                       'cbar_trueon': label of colorbar when mtrue is shown}
                        (ii) for multiple series J>1:
                        (ii.1) scalar x: {'y_j': label of the 'j'th y}, no separate labels for mtrue or confidint
                        (ii.2) 2-vector x: {'cbar_trueoff_j': label of colorbar when mtrue is not shown,
                        'cbar_trueon': label of colorbar when mtrue is shown}
   :param subplottitle: dictionary: {'y_j' title of the subplot of the jth series}. Only for 2-vector case, when
                        there are multiple series
   :param seriesperrow: integer, controls the how many series are shown in a row if there are multiple series.
                        It must be such that when the number of series is divided with it the result is integer
   :param savemode: if True figure is saved in the current working directory, requires filename to be given
   :param filepath: name under which  figure is saved as r'destination\filename' without file extension
   :param viewmode: if True the figure is displayed, blocking the execution

   RETURNS
   -------
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
    plt.rc('font', family='serif')

    # Clear previous figure
    plt.close()
    # Check if bivariate
    is_bivariate = isinstance(xdomain, list)
    # Get the number of series if J>1
    if ismultiple:
        J = len(mhat)

    # Set dafault values of fontsize, color, alpha, truestyle, legendlabel
    # fontsize
    if fontsize is None:
        fontsize = dict({'title': 12, 'xlabel': 11, 'ylabel': 11,
                         'xticklabel': 10, 'yticklabel': 10, 'legend': 10})
    # color, alpha, legendlabel
    # J=1 single series
    if not ismultiple:
        # Univariate
        if not is_bivariate:
            if color is None:
                color = dict({'mhat': 'b', 'confidintfill': 'royalblue', 'mtrue': 'red'})
            if alpha is None:
                alpha = dict({'mhat': 1, 'confidintfill': 0.6, 'mtrue': 1})
            if legendlabel is None:
                legendlabel = dict({'mhat': 'Estimated', 'mtrue': 'True', 'confidint': 'CI'})
            if truestyle is None:
                truestyle = '--'
        # Multivariate
        else:
            if color is None:
                color = 'plasma'
            if legendlabel is None:
                legendlabel = dict({'cbar_trueon': '$y$, $\hat{m}$(\\boldmath{$x$})',
                                    'cbar_trueoff': '$\hat{m}$(\\boldmath{$x$})'})
    # J>1 multiple series
    else:
        # Univariate
        if not is_bivariate:
            # Color
            # List of colors for the distributions. Use the same color with different alphas and linestyle
            # to visualise confidint and mtrue
            if color is None:
                color = dict()
                for j in range(1, J + 1):
                    actualcolor = tuple(np.random.rand(3))
                    color['mhat_' + str(j)] = actualcolor
                    color['confidintfill_' + str(j)] = actualcolor
                    color['mtrue_' + str(j)] = actualcolor
            # Alpha
            if alpha is None:
                alpha = dict()
                for j in range(1, J + 1):
                    alpha['mhat_' + str(j)] = 1
                    alpha['confidintfill_' + str(j)] = 0.6
                    alpha['mtrue_' + str(j)] = 1
            # Legend
            if legendlabel is None:
                legendlabel = dict()
                for j in range(1, J + 1):
                        legendlabel['y_' + str(j)] = 'Series ' + str(j)
            if truestyle is None:
                truestyle = '--'
        # Multivariate
        else:
            if color is None:
                color = 'plasma'
            if legendlabel is None:
                legendlabel = dict()
                for j in range(1, J + 1):
                    legendlabel['cbar_trueon_' + str(j)] = '$y$, $\hat{m}$(\\boldmath{$x$})'
                    legendlabel['cbar_trueoff_' + str(j)] = '$\hat{m}$(\\boldmath{$x$})'
            if subplottitle is None:
                subplottitle = dict()
                for j in range(1, J + 1):
                    subplottitle['y_' + str(j)] = 'Series ' + str(j)

    # Make the basic plot
    # Univariate
    if not is_bivariate:
        f, ax = plt.subplots()
        # Add true density if required
        # J=1 single series
        if not ismultiple:
            if mtrueon:
                if mtrue is None:
                    raise Exception('If mtrueon=True mtrue is required')
                else:
                    ax.plot(xdomain, mtrue, truestyle, label=legendlabel['mtrue'], color=color['mtrue'],
                            alpha=alpha['mtrue'])
            # Estimated series
            ax.plot(xdomain, mhat, label=legendlabel['mhat'], color=color['mhat'], alpha=alpha['mhat'])
            # Add confidence intervals if required
            if confidinton:
                if confidint is None:
                    raise Exception('If confidinton=True, confidint is required.')
                else:
                    ax.fill_between(xdomain, confidint['ci_low'], confidint['ci_high'],
                                    label=legendlabel['confidint'],
                                    facecolor=color['confidintfill'], alpha=alpha['confidintfill'])

        # J>1 multiple series
        else:
            # Checking and setting mtrueon and confidinton
            # mtrueon
            # check if one boolean or list, set to J-long list if one boolean
            # one boolean
            if isinstance(mtrueon, bool):
                # error if mtruon is True but no mtrue is given
                if mtrueon and mtrue is None:
                    raise Exception('For mtrueon=True, mtrue must be given.\n')
                # error if one boolean mtrueon=True is given but mtrue is only one array
                if mtrueon and np.ndim(mtrue) != J:
                    raise Exception(['If ftruenon=True scalar boolean, mtrue must be an array of arrays of length J' +
                                    ' with corresponding arrays belonging to the \'j\'th distribution'][0])
                # list
                ftrueon_list = [mtrueon] * J
            # list of boolean
            else:
                # error if the number of Trues in mtrueon list is not the same as the number of arrays in mtrue
                num_ftrueon = sum([int(b) for b in mtrueon])
                if num_ftrueon != len(mtrue):
                    raise Exception('The number of Trues in mtrueon doesn\'t match the number of arrays in mtrue.\n')
                ftrueon_list = mtrueon
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
                # error if the number of Trues in mtrueon list is not the same as the number of arrays in mtrue
                num_confidinton = sum([int(b) for b in confidinton])
                if num_confidinton != len(confidint) / 2:
                    raise Exception(['The number of Trues in confidinton doesn\'t match the number of ci_low, ci_high' +
                     ' pairs in confidint.\n'][0])
                confidinton_list = confidinton

            # Plotting
            for j in range(1, J+1):
                if ftrueon_list[j-1]:
                    ax.plot(xdomain, mtrue[j - 1], truestyle,
                            color=color['mtrue_' + str(j)], alpha=alpha['mtrue_' + str(j)])
                # Estimated density
                ax.plot(xdomain, mhat[j - 1], label=legendlabel['y_' + str(j)],
                        color=color['mhat_' + str(j)], alpha=alpha['mhat_' + str(j)])
                # Add confidence intervals if required
                if confidinton_list[j-1]:
                        ax.fill_between(xdomain, confidint['ci_low_' + str(j)], confidint['ci_high_' + str(j)],
                                        facecolor=color['confidintfill_' + str(j)],
                                        alpha=alpha['confidintfill_' + str(j)])
        # Add legend if required
        if legendon:
            ax.legend(fontsize=fontsize['legend'])
        ax.set_title(plottitle, size=fontsize['title'])
        plt.setp(ax.get_xticklabels(), size=fontsize['xticklabel'], ha='center')
        plt.setp(ax.get_yticklabels(), size=fontsize['yticklabel'])
        plt.xlabel(xlabel, size=fontsize['xlabel'])
        plt.ylabel(ylabel, size=fontsize['ylabel'])

    # Bivariate
    else:
        # X's
        x1, x2 = xdomain[0], xdomain[1]
        # Alpha for dots
        alphadot = 0.7
        # Single series
        if not ismultiple:
            # Not showing true series
            if not mtrueon:
                f, ax = plt.subplots()
                # estimated
                axx = ax.tripcolor(x1, x2, mhat, cmap=color)
                ax.plot(x1, x2, 'ko', alpha=alphadot)
                ax.set_title(plottitle, size=fontsize['title'])
                plt.setp(ax.get_xticklabels(), size=fontsize['xticklabel'], ha='center')
                plt.setp(ax.get_yticklabels(), size=fontsize['yticklabel'])
                plt.xlabel(xlabel, size=fontsize['xlabel'])
                plt.ylabel(ylabel, size=fontsize['ylabel'])
                f.colorbar(axx, label=legendlabel['cbar_treuoff'])
            # Showing true series
            else:
                if mtrue is None:
                    raise Exception('If mtrueon=True mtrue is required')
                else:
                    f, ax = plt.subplots(1, 2, sharex=True, sharey=True)
                    # estimated
                    axx0 = ax[0].tripcolor(x1, x2, mhat, cmap=color)
                    ax[0].plot(x1, x2, 'ko', alpha=alphadot)
                    ax[0].set_title('Estimated', size=fontsize['title']-1)
                    plt.setp(ax[0].get_xticklabels(), size=fontsize['xticklabel'], ha='center')
                    plt.setp(ax[0].get_yticklabels(), size=fontsize['yticklabel'])
                    ax[0].set_ylabel(ylabel, size=fontsize['ylabel'])
                    ax[0].set_xlabel(xlabel, size=fontsize['xlabel'])
                    plt.colorbar(axx0, ax=ax[0])
                    # true
                    axx1 = ax[1].tripcolor(x1, x2, mtrue, cmap=color)
                    ax[1].plot(x1, x2, 'ko', alpha=alphadot)
                    ax[1].set_title('True', size=fontsize['title']-1)
                    plt.setp(ax[1].get_xticklabels(), size=fontsize['xticklabel'], ha='center')
                    plt.setp(ax[1].get_yticklabels(), size=fontsize['yticklabel'])
                    ax[1].set_xlabel(xlabel, size=fontsize['xlabel'])
                    f.colorbar(axx1, label=legendlabel['cbar_trueon'])
                    plt.suptitle(plottitle, size=fontsize['title'])
        # Multiple series
        else:
            # error if multiple series and mtrueon is not a single boolean
            if not isinstance(mtrueon, bool):
                raise Exception('For multiple series, if x is 2-vector, mtrueon must be a single boolean')
            # number of series per row
            if seriesperrow is None:
                seriesperrow = 1
            # Not showing true series
            if not mtrueon:
                f, ax_arr = plt.subplots(nrows=int(J / seriesperrow), ncols=seriesperrow)
                ax_arr = ax_arr.ravel()
                for j in range(0, J):
                    # estimated
                    axx = ax_arr[j].tripcolor(x1, x2, mhat[j], cmap=color)
                    ax_arr[j].plot(x1, x2, 'ko', alpha=alphadot)
                    ax_arr[j].set_title(subplottitle['y_' + str(j + 1)], size=fontsize['title'] - 1)
                    plt.setp(ax_arr[j].get_xticklabels(), size=fontsize['xticklabel'], ha='center')
                    plt.setp(ax_arr[j].get_yticklabels(), size=fontsize['yticklabel'])
                    # set xticklabel only in the last row
                    if (J - j) <= seriesperrow:
                        ax_arr[j].set_xlabel(xlabel, size=fontsize['xlabel'])
                    # set yticklabel only for the first column in the row
                    if j % seriesperrow == 0:
                        ax_arr[j].set_ylabel(ylabel, size=fontsize['ylabel'])
                    # set cbarlabel only in the last column in the row
                    if j % seriesperrow == (seriesperrow -1):
                        f.colorbar(axx, ax=ax_arr[j], label=legendlabel['cbar_trueoff_' + str(j + 1)])
                    else:
                        f.colorbar(axx, ax=ax_arr[j])
                plt.suptitle(plottitle, size=fontsize['title'])
                plt.tight_layout()
                plt.subplots_adjust(top=0.85)
            # Showing true series
            else:
                f, ax_arr = plt.subplots(nrows=int(J / seriesperrow), ncols=seriesperrow * 2)
                ax_arr = ax_arr.ravel()
                for j in range(0, 2 * J):
                    # when j is even, plot the estimated
                    if j % 2 == 0:
                        axxE = ax_arr[j].tripcolor(x1, x2, mhat[int(j / 2)], cmap=color)
                        ax_arr[j].plot(x1,x2, 'ko', alpha=alphadot)
                        ax_arr[j].set_title(subplottitle['y_' + str(int(j / 2 + 1))] + ' -- estimated',
                                            size=fontsize['title'] - 1)
                        plt.setp(ax_arr[j].get_xticklabels(), size=fontsize['xticklabel'], ha='center')
                        plt.setp(ax_arr[j].get_yticklabels(), size=fontsize['yticklabel'])
                        # set xticklabel only in the last row
                        if (2 * J - j) <= seriesperrow * 2:
                            ax_arr[j].set_xlabel(xlabel, size=fontsize['xlabel'])
                        # when at beginning of new line set ytick
                        if j % (seriesperrow * 2) == 0:
                            ax_arr[j].set_ylabel(ylabel, size=fontsize['ylabel'])
                        f.colorbar(axxE, ax=ax_arr[j])
                    # when j is odd, plot the true
                    else:
                        axxT = ax_arr[j].tripcolor(x1, x2, mtrue[int((j - 1) / 2)], cmap=color)
                        ax_arr[j].plot(x1, x2, 'ko', alpha=alphadot)
                        ax_arr[j].set_title(subplottitle['y_' + str(int((j - 1) / 2 + 1))] + ' -- true',
                                            size=fontsize['title'] - 1)
                        plt.setp(ax_arr[j].get_xticklabels(), size=fontsize['xticklabel'], ha='center')
                        plt.setp(ax_arr[j].get_yticklabels(), size=fontsize['yticklabel'])
                        # set xticklabel only in the last row
                        if (2 * J - j) <= seriesperrow * 2:
                            ax_arr[j].set_xlabel(xlabel, size=fontsize['xlabel'])
                        # when at the end of line set cbar label
                        if j % (seriesperrow * 2) == (seriesperrow * 2 -1):
                            f.colorbar(axxT, ax=ax_arr[j],
                                       label=legendlabel['cbar_trueon_' + str(int((j - 1) / 2 + 1))])
                        else:
                            f.colorbar(axxT, ax=ax_arr[j])
                plt.suptitle(plottitle, size=fontsize['title'])
                plt.tight_layout()
                plt.subplots_adjust(top=0.85)
    # Save figure if required
    if savemode:
        f.set_size_inches(figsize(1.5)[0], figsize(0.9)[1])
        # save to pgf
        plt.savefig(filepath+'.pgf', bbox_inches='tight')
        # save to png
        plt.savefig(filepath+'.png', bbox_inches='tight')

    # Show figure if reguired
    if viewmode:
        plt.show(block=viewmode)
