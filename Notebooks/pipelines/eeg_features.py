"""
EEG feature extract functions
"""

import pywt
import math
import nolds
from pyentrp import entropy as ent
from scipy.stats import skew
from scipy.stats import kurtosis
import scipy
import scipy.signal
import scipy.stats as sp
import antropy as ant
from numba import njit
import numpy as np


"""
Feature level 1
"""


def min_X(X):
    return np.min(X)


def max_X(X):
    return np.max(X)


def std_X(X):
    return np.std(X)


def mean_X(X):
    return np.mean(X)


def median_X(X):
    return np.median(X)


def var_X(X):
    return np.var(X)


def coefficient_variation(X):
    return np.std(X)/np.mean(X)


def mean_absolute_X(X):
    return np.mean(np.abs(X))


def average_amplitude_change(X):
    n = len(X)
    y = 0
    for i in range(0,n-1):
        y += np.abs(X[i+1]-X[i])
    
    AAC = y/n
    return AAC


def cardinality(X, thres=0.01):

    n = len(X)
    y = X.tolist()
    y.sort()

    z = np.zeros(n-1)
    for i in range(0, n-1):
        if np.abs(y[i]-y[i+1]) > thres:
            z[i] = 1
    CARD = np.sum(z)

    return CARD


def enhanced_mean_absolute(X):
    l = len(X)
    y = 0

    for i in range(0, l):
        if 0.2*l <= i <= 0.8*l:
            p = 0.75
        else:
            p = 0.5
        y += np.abs((X[i])**p)
    EMAV = y/l
    return EMAV


def mean_amplitude_power(X):
    return np.linalg.norm(X, ord=2)**2/len(X)


def waveform_length(X):
    n = len(X)
    WL = 0
    for i in range(1,n):
        WL += np.abs(X[i]-X[i-1])
    
    return WL


def enhanced_wave_length(X):
    l = len(X)
    EML = 0
    for i in range(1, l):
        if i >= 0.2*l and i <= 0.8*l:
            p = 0.75
        else:
            p = 0.5
        EML += np.abs((X[i]-X[i-1])**p)
    
    return EML


def mean_curve_length(X):
    n = len(X)
    y = 0
    for i in range(1, n):
        y += abs(X[i]-X[i-1])
    return y/n


def peak_X(X):
    Peak = np.max([np.abs(max_X(X)), np.abs(min_X(X))])
    return Peak


def num_zero_crossing(X):
    return ant.num_zerocross(X)


"""
Feature level 2
"""


def totalVariation(X):
    Max = np.max(X)
    Min = np.min(X)
    return np.sum(np.abs(np.diff(X))) / ((Max - Min) * (len(X) - 1))


def log_root_sum_of_sequential_Variation(X):
    n = len(X)
    y = np.zeros(n - 1)
    for i in range(1, n):
        y[i - 1] = (X[i] - X[i - 1]) ** 2

    return np.log10(math.sqrt(np.sum(y)))


def first_order_diff(X):
    sum_diff = np.sum(np.diff(X))
    min_diff = np.min(np.diff(X))
    max_diff = np.max(np.diff(X))
    mean_diff = np.max(np.diff(X))
    median_diff = np.median(np.diff(X))
    return sum_diff, min_diff, max_diff, mean_diff, median_diff


def second_order_diff(X):
    sum_diff = np.sum(np.diff(np.diff(X)))
    min_diff = np.min(np.diff(np.diff(X)))
    max_diff = np.max(np.diff(np.diff(X)))
    mean_diff = np.max(np.diff(np.diff(X)))
    median_diff = np.median(np.diff(np.diff(X)))
    return sum_diff, min_diff, max_diff, mean_diff, median_diff


# Skewness is a measure of whether a distribution is symmetric.
# The normal distribution is symmetrical on the left and right, and the skewness coefficient is 0.
# Larger positive values indicate that the distribution has longer tails on the right.
# Larger negative values indicate longer left tails
def skew_X(X):
    skewness = skew(X)
    return skewness


# The kurtosis coefficient is used to measure the degree to which the data is clustered in the center.
# In the case of a normal distribution, the kurtosis coefficient value is 3.
# A kurtosis coefficient >3 indicates that
# the observations are more concentrated and have shorter tails than the normal distribution;
# A kurtosis coefficient of <3 indicates that the observations are less concentrated
# and have longer tails than a normal distribution, similar to a rectangular uniform distribution.
def kurs_X(X):
    kurs = kurtosis(X)
    return kurs


# RMS value is often used to analyze noise
def rms_X(X):
    RMS = np.sqrt((np.sum(np.square(X))) * 1.0 / len(X))
    return RMS


# Peak-to-average ratio
def papr_X(X):
    Peak = peak_X(X)
    RMS = rms_X(X)
    PAPR = np.square(Peak) * 1.0 / np.square(RMS)
    return PAPR


# Hurst exponent
def Hurst(X):
    y = nolds.hurst_rs(X)
    return y


# compute Petrosian's Fractal Dimension value
def Petrosian_FD(X):
    D = np.diff(X)

    delta = 0
    N = len(X)
    # number of sign changes in signal
    for i in range(1, len(D)):
        if D[i] * D[i - 1] < 0:
            delta += 1

    feature = np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * delta)))

    return feature  # ant.petrosian_fd(X)


# Hjorth Parameter: activity, mobility and complexity
def Hjorth(X):
    D = np.diff(X)
    D = list(D)
    D.insert(0, X[0])
    VarX = np.var(X)
    VarD = np.var(D)
    Activity = VarX
    Mobility = np.sqrt(VarD / VarX)

    DD = np.diff(D)
    VarDD = np.var(DD)
    Complexity = np.sqrt(VarDD / VarD) / Mobility

    return Activity, Mobility, Complexity


# Katz fractal dimension
def KFD(X):
    return ant.katz_fd(X)


# Higuchi fractal dimension
def HFD(X):
    return ant.higuchi_fd(X)


# Detrended fluctuation analysis
def DFA(X):
    return ant.detrended_fluctuation(X), nolds.dfa(X)


"""
Feature level 3
"""



def signal_energy(X):
    return np.linalg.norm(X, ord=2)**2


def mean_energy(X):
    return np.mean(X**2)


def mean_teager_energy(X):
    n = len(X)
    y = 0
    for i in range(2, n):
        y += (X[i-1]**2) - X[i] * X[i-2]
    return y/n


def log_energy_entropy(X):
    return np.sum(np.log(X**2))


def renyi_entropy(X, alpha=2):

    P =  X**2 / np.sum(X**2)
    En = P**alpha
    return (1-(1-alpha)) * np.log2(np.sum(En))


def wavelet_entopy(X):
    [pEA5, pED5, pED4, pED3, pED2, pED1] = relativePower(X)
    wavelet_entopy = - (pEA5*math.log(pEA5) + pED5*math.log(pED5)
    + pED4*math.log(pED4) + pED3*math.log(pED3) + pED2*math.log(pED2) + pED1*math.log(pED1))
    return wavelet_entopy


def sample_entropy(X):
    # ant.sample_entropy(x)
    y = nolds.sampen(X)
    return y


# A method to measure the complexity of a time series, the size of the permutation entropy H represents the randomness
# of the time series. The smaller the value, the more regular the time series is. On the contrary,
# the more random the time series is.
# def ant_permutation_entropy(X):
#     return ant.perm_entropy(X, normalize=True)
def permutation_entropy(X):
    y = ent.permutation_entropy(X, 4, 1)
    return y, ant.perm_entropy(X, normalize=True)


def shannon_entropy(X):
    y = ent.shannon_entropy(X)
    return y


def spectral_entropy(X):
    return ant.spectral_entropy(X, sf=125, method='welch', normalize=True)


def svd_entropy(X):
    return ant.svd_entropy(X, normalize=True)


def approximate_entropy(X):
    return ant.app_entropy(X)


"""
complex function
"""


# Lempel-Ziv Complexity
# This one running time is too long
# def LZC(X):
#     return ant.lziv_complexity(X, normalize=True)
def LZC(data, threshold = None):
    """
    Returns the Lempel-Ziv Complexity (LZ76) of the given data.

    Parameters
    ----------
    data: array_like
        The signal.
    theshold: numeric, optional
        A number use to binarize the signal. The values of the signal above
        threshold will be converted to 1 and the rest to 0. By default, the
        median of the data.

    References
    ----------
    .. [1] M. Aboy, R. Hornero, D. Abasolo and D. Alvarez, "Interpretation of
           the Lempel-Ziv Complexity Measure in the Context of Biomedical
           Signal Analysis," in IEEE Transactions on Biomedical Engineering,
           vol. 53, no.11, pp. 2282-2288, Nov. 2006.
    """
    if not threshold:
        threshold=np.median(data)

    n = len(data)

    sequence = _binarize(data, threshold)

    c = _LZC(sequence)
    b = n/np.log2(n)

    lzc = c/b

    return lzc


@njit
def _LZC(sequence):# pragma: no cover
    n = len(sequence)
    complexity = 1

    q0    = 1
    qSize = 1

    sqi   = 0
    where = 0

    while q0 + qSize <= n:
        # If we are checking the end of the sequence we just need to look at
        # the last element
        if sqi != q0-1:
            contained, where = _isSubsequenceContained(sequence[q0:q0+qSize],
                                                    sequence[sqi:q0+qSize-1])
        else:
            contained = sequence[q0+qSize] == sequence[q0+qSize-1]

         #If Q is contained in sq~, we increase the size of q
        if contained:
            qSize+=1
            sqi = where
        #If Q is not contained the complexity is increased by 1 and reset Q
        else:
            q0+=qSize
            qSize=1
            complexity+=1
            sqi=0

    return complexity


def _binarize(data, threshold):
    if  not isinstance(data, np.ndarray):
        data = np.array(data)

    return np.array(data > threshold, np.uint8)


@njit
def _isSubsequenceContained(subSequence, sequence):# pragma: no cover
    """
    Checks if the subSequence is into the sequence and returns a tuple that
    informs if the subsequence is into and where. Return examples: (True, 7),
    (False, -1).
    """
    n = len(sequence)
    m = len(subSequence)

    for i in range(n-m+1):
        equal = True
        for j in range(m):
            equal = subSequence[j] == sequence[i+j]
            if not equal:
                break

        if equal:
            return True, i

    return False, -1


def embed_seq(time_series, tau, embedding_dimension):
    """Build a set of embedding sequences from given time series `time_series`
    with lag `tau` and embedding dimension `embedding_dimension`.

    Let time_series = [x(1), x(2), ... , x(N)], then for each i such that
    1 < i <  N - (embedding_dimension - 1) * tau,
    we build an embedding sequence,
    Y(i) = [x(i), x(i + tau), ... , x(i + (embedding_dimension - 1) * tau)].

    All embedding sequences are placed in a matrix Y.

    Parameters
    ----------

    time_series
        list or numpy.ndarray

        a time series

    tau
        integer

        the lag or delay when building embedding sequence

    embedding_dimension
        integer

        the embedding dimension

    Returns
    -------

    Y
        2-embedding_dimension list

        embedding matrix built

    Examples
    ---------------
    >>> import pyeeg
    >>> a=range(0,9)
    >>> pyeeg.embed_seq(a,1,4)
    array([[0,  1,  2,  3],
           [1,  2,  3,  4],
           [2,  3,  4,  5],
           [3,  4,  5,  6],
           [4,  5,  6,  7],
           [5,  6,  7,  8]])
    >>> pyeeg.embed_seq(a,2,3)
    array([[0,  2,  4],
           [1,  3,  5],
           [2,  4,  6],
           [3,  5,  7],
           [4,  6,  8]])
    >>> pyeeg.embed_seq(a,4,1)
    array([[0],
           [1],
           [2],
           [3],
           [4],
           [5],
           [6],
           [7],
           [8]])

    """
    if not type(time_series) == np.ndarray:
        typed_time_series = np.asarray(time_series)
    else:
        typed_time_series = time_series

    shape = (
        typed_time_series.size - tau * (embedding_dimension - 1),
        embedding_dimension
    )

    strides = (typed_time_series.itemsize, tau * typed_time_series.itemsize)

    return np.lib.stride_tricks.as_strided(
        typed_time_series,
        shape=shape,
        strides=strides
    )


def LLE(x, tau, n, T=1/125, fs=125):
    """Calculate largest Lyauponov exponent of a given time series x using
    Rosenstein algorithm.

    Parameters
    ----------

    x list  a time series

    n integer embedding dimension

    tau integer Embedding lag

    fs  integer Sampling frequency

    T   integer  Mean period

    Returns
    ----------

    Lexp  Largest Lyapunov Exponent

    Notes
    ----------
    A n-dimensional trajectory is first reconstructed from the observed data by
    use of embedding delay of tau, using pyeeg function, embed_seq(x, tau, n).
    Algorithm then searches for nearest neighbour of each point on the
    reconstructed trajectory; temporal separation of nearest neighbours must be
    greater than mean period of the time series: the mean period can be
    estimated as the reciprocal of the mean frequency in power spectrum

    Each pair of nearest neighbours is assumed to diverge exponentially at a
    rate given by largest Lyapunov exponent. Now having a collection of
    neighbours, a least square fit to the average exponential divergence is
    calculated. The slope of this line gives an accurate estimate of the
    largest Lyapunov exponent.

    References
    ----------
    Rosenstein, Michael T., James J. Collins, and Carlo J. De Luca. "A
    practical method for calculating largest Lyapunov exponents from small data
    sets." Physica D: Nonlinear Phenomena 65.1 (1993): 117-134.

    """

    # from embedded_sequence import embed_seq

    Em = x #embed_seq(x, tau, n)
    Em = embed_seq(x, tau, n)
    M = len(Em)
    A = np.tile(Em, (len(Em), 1, 1))
    B = np.transpose(A, [1, 0, 2])

    #  square_dists[i,j,k] = (Em[i][k]-Em[j][k])^2
    square_dists = (A - B) ** 2

    #  D[i,j] = ||Em[i]-Em[j]||_2
    D = np.sqrt(square_dists[:, :, :].sum(axis=2))

    # Exclude elements within T of the diagonal
    band = np.tri(D.shape[0], k=T) - np.tri(D.shape[0], k=-T - 1)
    band[band == 1] = np.inf

    # nearest neighbors more than T steps away
    neighbors = (D + band).argmin(axis=0)

    # in_bounds[i,j] = (i+j <= M-1 and i+neighbors[j] <= M-1)
    inc = np.tile(np.arange(M), (M, 1))
    row_inds = (np.tile(np.arange(M), (M, 1)).T + inc)
    col_inds = (np.tile(neighbors, (M, 1)) + inc.T)
    in_bounds = np.logical_and(row_inds <= M - 1, col_inds <= M - 1)

    # Uncomment for old (miscounted) version
    # in_bounds = numpy.logical_and(row_inds < M - 1, col_inds < M - 1)
    row_inds[~in_bounds] = 0
    col_inds[~in_bounds] = 0

    # neighbor_dists[i,j] = ||Em[i+j]-Em[i+neighbors[j]]||_2
    neighbor_dists = np.ma.MaskedArray(D[row_inds, col_inds], ~in_bounds)

    #  number of in-bounds indices by row
    J = (~neighbor_dists.mask).sum(axis=1)

    # Set invalid (zero) values to 1; log(1) = 0 so sum is unchanged
    neighbor_dists[neighbor_dists == 0] = 1
    d_ij = np.sum(np.log(neighbor_dists.data), axis=1)
    mean_d = d_ij[J > 0] / J[J > 0]

    x = np.arange(len(mean_d))
    X = np.vstack((x, np.ones(len(mean_d)))).T
    [m, c] = np.linalg.lstsq(X, mean_d)[0]
    Lexp = fs * m
    return Lexp


# non linear
# ?If the sampling frequency is 125Hz, the maximum frequency of the signal is 62.5Hz,
# and 5/8 layer wavelet decomposition is performed.
def relativePower(X):
    Ca5, Cd5, Cd4, Cd3, Cd2, Cd1 = pywt.wavedec(X, wavelet='db4', level=5)
    EA5 = sum([i*i for i in Ca5])
    ED5 = sum([i*i for i in Cd5])
    ED4 = sum([i*i for i in Cd4])
    ED3 = sum([i*i for i in Cd3])
    ED2 = sum([i*i for i in Cd2])
    ED1 = sum([i*i for i in Cd1])
    E = EA5 + ED5 + ED4 + ED3 + ED2 + ED1
    pEA5 = EA5/E
    pED5 = ED5/E
    pED4 = ED4/E
    pED3 = ED3/E
    pED2 = ED2/E
    pED1 = ED1/E
    return pEA5, pED5, pED4, pED3, pED2, pED1



