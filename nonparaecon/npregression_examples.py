# This file demonstrates the use of functions on npregresssion.py: Nadaraya-Watson estimator,
#  local polynomial regression and plotting tools

# Import dependencies
import numpy as np
from nonparaecon import npregression

######################################### GENERATE DATA ############################################################
print('\n\n ##############################################\n GENERATING DATA \n ###################################\n')

# generate x
# scalar
x = np.arange(-5, 10, 0.04)
# multi
x1, x2 = np.random.uniform(low=-5, high=5, size=100), np.random.uniform(low=-5, high=10, size=100)
X = np.matrix([x1, x2]).T
# generate y
# data generating processes as functions


def datagenproc(x_in):
    y = 20 + x_in ** 2 * np.cos(0.5 * x_in) * int(x_in < 0) + 10 * np.sin(x_in) * np.cos(x_in) * x_in * int(0 <= x_in)
    return y


def datagenproc2(x1_in, x2_in):
    y_out = x1_in * x2_in
    return y_out


# generate y and add noise
# scalar x
y = [datagenproc(x_in=i) for i in x]
ynoise = [datagenproc(x_in=i) + 20 * np.random.randn(1)[0] for i in x]
# multiple x
y1 = [datagenproc2(i[0, 0], i[0, 1]) for i in X]
ynoise1 = [datagenproc2(i[0, 0], i[0, 1]) + 10 * np.random.randn(1)[0] for i in X]

######################################################### ESTIMATION ############################################
print('\n\n ################################################\n ESTIMATION \n ###################################\n')

# Call the Nadaraya-Watson estimator and the local polynomial regression function, obtain standard errors for the latter

# scalar x
# Nadaraya-Watson
print('Working on scalar x, Nadaraya-Watson\n')
yhat_nw = [npregression.npregression_nadwats(x=i, xdata=x, ydata=ynoise, kerneltype='triangular', bandwidth=0.6)
           for i in x]
# local polynomial regression
polorder = 3
print('Working on scalar x, local polynomial regression, degree', polorder, '\n')
# do cross validation to get the optimal bandwidth
h_opt, sse = npregression.npregression_locpollscv(searchrange=np.arange(2, 5, 0.5), xdata=x,
                                                  ydata=ynoise,
                                                  poldegree=polorder, kerneltype='triangular', subsamplesize=None,
                                                  get_sse=True)
print('LSCV results: \nh_opt=\n', h_opt, '\nsse=\n', sse, '\n\n')
# pre-allocation
yhat, se, ci_low, ci_high = list(), list(), list(), list()
for i in x:
    # compute mhat, SE, and CI for each data point
    actual_yhat, actual_se, actual_ci = npregression.npregression_locpol(x=i, xdata=x,
                                                                         ydata=ynoise,
                                                                         poldegree=polorder,
                                                                         kerneltype='triangular',
                                                                         bandwidth=h_opt,
                                                                         get_se=True,
                                                                         get_ci=True)
    # append to list
    yhat.append(actual_yhat)
    se.append(actual_se)
    ci_low.append(actual_ci[0])
    ci_high.append(actual_ci[1])

# multiple x
# Nadaraya-Watson
print('\n\nWorking on multiple x, Nadaraya-Watson\n')
yhat_nw1 = [npregression.npregression_nadwats(x=np.array([i[0, 0], i[0, 1]]), xdata=np.array(X), ydata=ynoise1,
                                              kerneltype='epanechnikov', bandwidth=0.6) for i in X]
# local polynomial regression
polorder = 3
print('\nWorking on multiple x, local polynomial regression, degree', polorder, '\n')
# do cross validation to get the optimal bandwidth
h_opt1, sse1 = npregression.npregression_locpollscv(searchrange=np.arange(4, 10, 0.5), xdata=X,
                                                    ydata=ynoise1, poldegree=polorder - 1,
                                                    kerneltype='triangular', subsamplesize=80,
                                                    get_sse=True)
print('LSCV results: \nh_opt1=\n', h_opt1, '\nsse1=\n', sse1)
# regression
yhat1 = [npregression.npregression_locpol(x=np.matrix([i[0, 0], i[0, 1]]), xdata=X, ydata=ynoise1,
                                          poldegree=polorder - 1, kerneltype='gaussian', bandwidth=h_opt1,
                                          get_ci=False, get_se=False) for i in X]


####################################################### VISUALISATION ###############################################

# Compare Nad-Wats and loc. pol. reg to each other
# scalar x
npregression.npregression_plot(mhat=[yhat_nw, yhat], ismultiple=True, xdomain=x,
                               plottitle='Nad-Wat estimates and local polynomial regression, scalar $x$, $p=$'
                                         + str(polorder),
                               xlabel='$x$', ylabel='$y, \hat{m}(x)$', mtrueon=True, mtrue=[y, y], truestyle='.',
                               confidinton=[False, True], confidint=dict({'ci_low_2': ci_low, 'ci_high_2': ci_high}),
                               legendlabel=dict({'y_1': 'Nad.-Wats.', 'y_2': 'Loc. Pol.'}),
                               seriesperrow=2, savemode=False,
                               filepath=r'C:\Users\Máté\Dropbox\CEU\2017 Spring\Nonparametric\proba',
                               viewmode=True)
# multiple
npregression.npregression_plot(mhat=[yhat_nw1, yhat1], ismultiple=True, xdomain=[x1, x2],
                               plottitle='Nad.-Wats. vs loc. pol. reg., d-vector $x$, $p=$' + str(polorder),
                               xlabel='$x_1$', ylabel='$x_2$', mtrueon=False, mtrue=y1, confidinton=True,
                               subplottitle=dict({'y_1': 'Nad.-Wats. ', 'y_2': 'Loc. Pol. Reg.'}),
                               savemode=False, filepath=r'C:\Users\Máté\Dropbox\CEU\2017 Spring\Nonparametric\proba',
                               viewmode=False)
# Compare to the true y values
npregression.npregression_plot(mhat=[yhat_nw1, yhat1], ismultiple=True, xdomain=[x1, x2],
                               plottitle='Nad.-Wats. vs loc. pol. reg., d-vector $x$, $p=$' + str(polorder),
                               xlabel='$x_1$', ylabel='$x_2$', mtrueon=True, mtrue=[y1, y1], confidinton=True,
                               subplottitle=dict({'y_1': 'Nad.-Wats. ', 'y_2': 'Loc. Pol. Reg.'}),
                               savemode=False, filepath=r'C:\Users\Máté\Dropbox\CEU\2017 Spring\Nonparametric\proba',
                               viewmode=True)
