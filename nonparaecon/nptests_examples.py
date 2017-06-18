# Examples demonstrating files in nptest.py

# Import dependencies
import numpy as np
from nonparaecon import kde, nptests

# Generate data
# x
x = np.random.uniform(low=-8, high=10, size=100)
# generate y
# data generating processes as functions


def datagenproc(x):
    #y = 20 + x ** 2 * np.cos(0.5 * x) * int(x < 0) + 10 * np.sin(x) * np.cos(x) * x * int(0 <= x)
    y = -x
    return y

y = [datagenproc(i) for i in x]
ynoise = [datagenproc(i) + 2 * np.random.randn(1)[0] for i in x]
#ynoise = np.random.randn(len(x))

# correlation coeff
print('r_xy=\n', np.corrcoef(np.array([x ,ynoise]).T, rowvar=0))

# get Silvermans opt bandwidth
h_x = kde.kde_pdf(x=None, sampledata=np.array(x), kerneltype='epanechnikov', biascorrected=False, getsilverman=True)
h_y = kde.kde_pdf(x=None, sampledata=np.array(y), kerneltype='epanechnikov', biascorrected=False, getsilverman=True)

# check independence
tstat = nptests.nptests_ahmadli(xdata=x, ydata=y, bandwidthx=h_x, bandwidthy=h_y, kerneltype='epanechnikov')
print('tstat=',tstat)