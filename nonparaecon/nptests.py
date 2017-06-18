def nptests_ahmadli(xdata, ydata, bandwidthx, bandwidthy, kerneltype, kernelorder=2,
                    subsamplesize=None, getpvalue=False):
    """
    Ahmad & Li nonparametrics test of statistical indenpendence of two random variables, which can be multivariate
    :param xdata: sample data on x, n*d_x sized, n: number of observations, d_x number of variables in x
    :param ydata: sample data on y, n*d_y sized, n: number of observations, d_y number of variables in y
    :param bandwidthx: scalar or list of length d_x, bandwidths to use in the kernel. If scalar, the same bandwidth
     is used for all variables in x
    :param bandwidthy: scalar or list of length d_y, bandwidths to use in the kernel. If scalar, the same bandwidth
     is used for all variables in y
    :param kerneltype: string, a name of kernel from kernels.py
    :param kernelorder: order of the kernel
    :param subsamplesize: if given the test uses only subsamplesize randomly chosen elements of the original sample
    :param getpvalue: if True, p-value is returned as well
    :return:
    """
    # Import dependencies
    import numpy as np
    from nonparaecon.kde import kde_pdf
    from nonparaecon import kernels

    # Subsample to use
    if not(subsamplesize is None):
        # seed random
        np.random.seed([0])
        # permute observations and keep required size
        permindex = np.random.permutation(len(ydata))
        xdata = xdata[permindex[0:subsamplesize]]
        ydata = ydata[permindex[0:subsamplesize]]

    # Get sizes
    n = len(xdata)
    try:
        d_x = np.size(xdata, 1)
    except:
        d_x = 1
    try:
        d_y = np.size(ydata, 1)
    except:
        d_y = 1
    #print('Perceived  number of variables in x and y respectively: ', d_x, d_y)

    # Joint array
    try:
        xyarray = np.array(np.concatenate([xdata, ydata], 1))
    except:
        xyarray = np.array([xdata, ydata]).T

    # Expand bandwidths to list if scalar
    if np.isscalar(bandwidthx):
        bandwidthx_exp = [bandwidthx] * d_x
    else:
        bandwidthx_exp = bandwidthx
    if np.isscalar(bandwidthy):
        bandwidthy_exp = [bandwidthy] * d_y
    else:
        bandwidthy_exp = bandwidthy
    # join
    bandwidthxy = bandwidthx_exp + bandwidthy_exp

    # Components of Itilde
    itilde1 = np.array([(n - 1) / n * kde_pdf(x=xyarray[i], sampledata=xyarray, kerneltype=kerneltype,
                                bandwidth=bandwidthxy, kernelorder=kernelorder, biascorrected=False,
                                leaveoneout=True, leftoutindex=i) for i in range(n)]).mean()
    itilde2 = np.array([(n - 1) / n * kde_pdf(x=xdata[i], sampledata=xdata, kerneltype=kerneltype,
                                bandwidth=bandwidthx, kernelorder=kernelorder,
                                biascorrected=False, leaveoneout=True, leftoutindex=i) * \
                        (n - 1) / n * kde_pdf(x=ydata[j], sampledata=ydata, kerneltype=kerneltype,
                                bandwidth=bandwidthy, kernelorder=kernelorder,
                                biascorrected=False, leaveoneout=True, leftoutindex=j)
                        for i in range(n) for j in range(n)]).mean()
    itilde3 = -2 * np.array([(n - 1) / n * kde_pdf(x=xdata[i], sampledata=xdata, kerneltype=kerneltype,
                                     bandwidth=bandwidthx, kernelorder=kernelorder, biascorrected=False,
                                     leaveoneout=True, leftoutindex=i) * \
                             (n - 1) / n * kde_pdf(x=ydata[i], sampledata=ydata, kerneltype=kerneltype,
                                     bandwidth=bandwidthy, kernelorder=kernelorder, biascorrected=False,
                                     leaveoneout=True, leftoutindex=i) for i in range(n)]).mean()

    # Numerator
    itilde = itilde1 + itilde2 + itilde3
    bandwidthproduct = np.product(np.array(bandwidthxy))
    numerator = bandwidthproduct * itilde

    # Denominator
    # k
    # get kernel
    kernel = getattr(kernels, kerneltype)
    # univariate x, univariate y
    if d_x == 1 and d_y == 1:
        k = np.array([kernel((xdata[i] - xdata[j]) / bandwidthx, kernelorder) ** 2 * \
                      kernel((ydata[i] - ydata[j]) / bandwidthy, kernelorder) ** 2
                      for j in range(n) for i in range(n) if j != i]).sum()
    # multivariate x, univariate y
    elif d_x > 1 and d_y == 1:
        k = np.array([np.product([kernel((xdata[i, d] - xdata[j, d]) / bandwidthx_exp[d], kernelorder)
                                  for d in range(d_x)]) ** 2 *\
                      kernel((ydata[i] - ydata[j]) / bandwidthy, kernelorder) ** 2
                      for j in range(n) for i in range(n) if j != i]).sum()
    # univariate x, multivariate y
    elif d_x == 1 and d_y > 1:
        k = np.array([kernel((xdata[i] - xdata[j]) / bandwidthx, kernelorder) ** 2 *\
                      np.product([kernel((ydata[i, d] - ydata[j, d]) / bandwidthy_exp[d], kernelorder)]
                                  for d in range(d_y)) ** 2
                      for j in range(n) for i in range(n) if j != i]).sum()
    # multivariate x, multivariate y
    else:
        k = np.array([np.product([kernel((xdata[i, d] - xdata[j, d]) / bandwidthx_exp[d], kernelorder)
                                  for d in range(d_x)]) ** 2 *\
                      np.product([kernel((ydata[i, d] - ydata[j, d]) / bandwidthy_exp[d], kernelorder)]
                                  for d in range(d_y)) ** 2
                      for j in range(n) for i in range(n) if j != i]).sum()

    denominator = np.sqrt(2 * k)
    # Test statistics
    teststat = n **2 * numerator / denominator
    if not getpvalue:
        return teststat
