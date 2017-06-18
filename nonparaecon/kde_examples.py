# This file demonstrates the use of kde_query kde_percentile and kde_plot in kde.py. Requires kernels.py.

# Import dependencies
import numpy as np
from scipy import stats
from nonparaecon import kde

# Create a random sample from Chi2(df=20) of size n=1000 and one from N(mu,sigma) of size=50
# Chi2(df=20)
sampleUni = np.random.chisquare(df=20, size=1000)
# N(mu, sigma)
mu = [0, 0]
sigma = [[1, 0.9], [0.9, 1]]
sampleMulti = np.random.multivariate_normal(mu, sigma, size=50) ** 2

# Use kde_pdf to predict the density based on these sample data in a range. Also predict confidence intervals for the
# univariate and compute the true density in the range.
support = np.arange(0, 40, 0.8)
supportx1 = np.arange(-4, 4, 0.2)
supportx2 = np.arange(-3, 3, 0.2)
# Predefine the lists
fhatUni = list()
ci_low = list()
ci_high = list()
fUni = list()
# Univariate
for x in support:
    # Call the function with CI mode
    hat, low, high = kde.kde_pdf(x, sampleUni, 'epanechnikov', kernelorder=2, biascorrected=True,
                                 correctboundarybias=False, flowbound=0, bandwidth=None, confidint=True)
    fhatUni.append(hat)
    ci_low.append(low)
    ci_high.append(high)
    # Compute the true density
    fUni.append(stats.chi2.pdf(x=x, df=20))
print('\nUnivariate estimation is done.\n')
# Multivariate
fhatMulti = [[kde.kde_pdf(np.array([x1, x2]), sampleMulti, 'epanechnikov',
                          correctboundarybias=False, flowbound=np.array([0, 0]),
                          bandwidth=np.array([0.5, 0.5]), confidint=True)
              for x1 in supportx1] for x2 in supportx2]
fMulti = [[stats.multivariate_normal.pdf([x1, x2], mean=mu, cov=sigma) for x1 in supportx1] for x2 in supportx2]
print('Multivariate estimation is done.\n')


# Call kde_plot to visualise
# Univariate
kde.kde_plot(fhat=[fhatUni, np.array(fhatUni) + 2], ismultiple=True, fsupport=support, plottitle='KDE',
             xlabel='$x$', ylabel='$f(x), \hat{f}(x)$', ftrueon=True, ftrue=[fUni, np.array(fUni) + 2],
             confidinton=[True, False], confidint=dict({'ci_low_1': ci_low, 'ci_high_1': ci_high}),
             #legendlabel=dict({'fhat': 'hat', 'ftrue': 'real', 'confidint': 'ci'}),
             savemode=False, filepath=r'C:\Users\Máté\Dropbox\CEU\2017 Spring\Nonparametric\proba', viewmode=True)
# Multivariate
kde.kde_plot(fhat=fhatMulti, ismultiple=False, fsupport=[supportx1, supportx2], plottitle='KDE',
             xlabel='$x_1$', ylabel='$x_2$', zlabel='$\hat{f}$({\\boldmath{$x$}})', ftrueon=False, ftrue=fMulti,
             savemode=True, filepath=r'C:\Users\Máté\Dropbox\CEU\2017 Spring\Nonparametric\proba2',viewmode=True)


# Now generate from Laplace(loc=0, scale=1) and estimate the 97.5% percentile based on the sample, using
# Epanechnikov kernel of order 4, with bandwidth=0.5.
sample2 = np.random.laplace(size=1000)
# True and Estimated percentile
percentile = stats.laplace.ppf(0.80)
percentilehat = kde.kde_quantile(alpha=0.80, sampledata=sample2,
                                 kerneltype='gaussian', kernelorder=2, bandwidth=None)
print('\nEstimated:', percentilehat, '\nTrue:', percentile, '\n')

# Call the estimated cdf for this laplace
Fhat = kde.kde_cdf(x=percentilehat, sampledata=sample2,
                   kerneltype='gaussian', kernelorder=2, bandwidth=None)
print('The estimated cdf at x='+str(percentilehat)+' is Fhat(x)=', Fhat, '\n',
      'The true is F(x)=', stats.laplace.cdf(percentilehat), '\n')

