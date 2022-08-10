# from spectrum import *
# import pyedflib

# from numba import njit


'''
计算时域10个bin特征
'''
# 过滤X中相等的点
# def filter_X(X):
#     X_new = []
#     length = np.shape(X)[0]
#     for i in range(1, length):
#         if i != 0 and X[i] == X[i-1]:
#             continue
#         X_new.append(X[i])
#     return X_new


# 求X中所有的极大值和极小值点
# def minmax_cal(X):
#     length = np.shape(X)[0]
#     min_value = []
#     min_index = []
#     max_value = []
#     max_index = []
#     first = ''
#     for i in range(1, length-1):
#         if X[i]<X[i-1] and X[i]<X[i+1]:
#             min_value.append(X[i])
#             min_index.append(i)
#         if X[i]>X[i-1] and X[i]>X[i+1]:
#             max_value.append(X[i])
#             max_index.append(i)
#     if len(min_index) and len(max_index):
#         if max_index[0] > min_index[0]:
#             first = 'min'
#         else:
#             first = 'max'
#         return min_value, max_value, first
#     else:
#         return None, None, None


# 计算所有的极大值和极小值的差值
# def minmax_sub_cal(X):
#     min_value, max_value, first = minmax_cal(X)
#     if min_value and max_value and first:
#         max_length = np.shape(max_value)[0]
#         sub = []
#         if first == 'min':
#             for i in range(max_length-1):
#                 sub.append(max_value[i] - min_value[i])
#                 sub.append(max_value[i] - min_value[i+1])
#         else:
#             for i in range(1, max_length-1):
#                 sub.append(max_value[i] - min_value[i-1])
#                 sub.append(max_value[i] - min_value[i])
#         return sub
#     else:
#         return None


# 计算极大极小值差值占比
# def minmax_percent_cal(X, step=10):
#     X = filter_X(X)
#     sub = minmax_sub_cal(X)
#     if sub:
#         length = int(np.shape(sub)[0])
#         max_value = max(sub)
#         min_value = min(sub)
#         diff = max_value - min_value
#         value = diff / step
#         nums = []
#         sub = np.array(sub)
#         for i in range(step):
#             scale_min = sub>=min_value+i*value
#             scale_max = sub<min_value+(i+1)*value
#             scale = scale_min & scale_max
#             num = np.where(scale)[0]
#             size = np.shape(num)[0]
#             nums.append(size)
#         nums[-1] = nums[-1] + sum(sub==max_value)
#         nums = np.array(nums, dtype = int)
#         per = nums / length
#         return per
#     else:
#         return [0,0,0,0,0,0,0,0,0,0]



# def autogressiveModelParameters(labels):
#     b_labels = len(labels)
#     feature = []
#     for i in range(14):
#         coeff, sig = alg.AR_est_YW(labels[i,:], 11,)
#         feature.append(coeff)
#     a = []
#     for i in range(11):
#         a.append(np.sum(feature[:][i])/14)
#
#     return a


# def autogressiveModelParametersBurg(labels):
#     feature = []
#     feature1 = []
#     model_order = 3
#     for i in range(14):
#         # spectrum
#         AR, rho, ref = arburg(labels[i], model_order)
#         feature.append(AR);
#     for j in range(14):
#         for i in range(model_order):
#             feature1.append(feature[j][i])
#
#     return feature1


# def wavelet_features(epoch):
#     cA_values = []
#     cD_values = []
#     cA_mean = []
#     cA_std = []
#     cA_Energy =[]
#     cD_mean = []
#     cD_std = []
#     cD_Energy = []
#     Entropy_D = []
#     Entropy_A = []
#     for i in range(14):
#         cA,cD=pywt.dwt(epoch[i,:],'coif1')
#         cA_values.append(cA)
#         cD_values.append(cD)		#calculating the coefficients of wavelet transform.
#     for x in range(14):
#         cA_mean.append(np.mean(cA_values[x]))
#         cA_std.append(np.std(cA_values[x]))
#         cA_Energy.append(np.sum(np.square(cA_values[x])))
#         cD_mean.append(np.mean(cD_values[x]))		# mean and standard deviation values of coefficents of each channel is stored .
#         cD_std.append(np.std(cD_values[x]))
#         cD_Energy.append(np.sum(np.square(cD_values[x])))
#         Entropy_D.append(np.sum(np.square(cD_values[x]) * np.log(np.square(cD_values[x]))))
#         Entropy_A.append(np.sum(np.square(cA_values[x]) * np.log(np.square(cA_values[x]))))
#     return np.sum(cA_mean)/14,np.sum(cA_std)/14,np.sum(cD_mean)/14,np.sum(cD_std)/14,np.sum(cA_Energy)/14,np.sum(cD_Energy)/14,np.sum(Entropy_A)/14,np.sum(Entropy_D)/14



#非线性特征提取
#采样频率为125Hz,则信号的最大频率为62.5Hz，进行5/8层小波分解
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


# def ap_entropy(X, M=10):
#     """Computer approximate entropy (ApEN) of series X, specified by M and R.
#
#     Suppose given time series is X = [x(1), x(2), ... , x(N)]. We first build
#     embedding matrix Em, of dimension (N-M+1)-by-M, such that the i-th row of
#     Em is x(i),x(i+1), ... , x(i+M-1). Hence, the embedding lag and dimension
#     are 1 and M-1 respectively. Such a matrix can be built by calling pyeeg
#     function as Em = embed_seq(X, 1, M). Then we build matrix Emp, whose only
#     difference with Em is that the length of each embedding sequence is M + 1
#
#     Denote the i-th and j-th row of Em as Em[i] and Em[j]. Their k-th elements
#     are Em[i][k] and Em[j][k] respectively. The distance between Em[i] and
#     Em[j] is defined as 1) the maximum difference of their corresponding scalar
#     components, thus, max(Em[i]-Em[j]), or 2) Euclidean distance. We say two
#     1-D vectors Em[i] and Em[j] *match* in *tolerance* R, if the distance
#     between them is no greater than R, thus, max(Em[i]-Em[j]) <= R. Mostly, the
#     value of R is defined as 20% - 30% of standard deviation of X.
#
#     Pick Em[i] as a template, for all j such that 0 < j < N - M + 1, we can
#     check whether Em[j] matches with Em[i]. Denote the number of Em[j],
#     which is in the range of Em[i], as k[i], which is the i-th element of the
#     vector k. The probability that a random row in Em matches Em[i] is
#     \simga_1^{N-M+1} k[i] / (N - M + 1), thus sum(k)/ (N - M + 1),
#     denoted as Cm[i].
#
#     We repeat the same process on Emp and obtained Cmp[i], but here 0<i<N-M
#     since the length of each sequence in Emp is M + 1.
#
#     The probability that any two embedding sequences in Em match is then
#     sum(Cm)/ (N - M +1 ). We define Phi_m = sum(log(Cm)) / (N - M + 1) and
#     Phi_mp = sum(log(Cmp)) / (N - M ).
#
#     And the ApEn is defined as Phi_m - Phi_mp.
#
#
#     Notes
#     -----
#     Please be aware that self-match is also counted in ApEn.
#
#     References
#     ----------
#     Costa M, Goldberger AL, Peng CK, Multiscale entropy analysis of biological
#     signals, Physical Review E, 71:021906, 2005
#
#     See also
#     --------
#     samp_entropy: sample entropy of a time series
#
#     """
#     R = int(np.std(X)/5)
#     N = len(X)
#
#     Em = embed_seq(X, 1, M)
#     A = np.tile(Em, (len(Em), 1, 1))
#     B = np.transpose(A, [1, 0, 2])
#     D = np.abs(A - B)  # D[i,j,k] = |Em[i][k] - Em[j][k]|
#     InRange = np.max(D, axis=2) <= R
#
#     # Probability that random M-sequences are in range
#     Cm = InRange.mean(axis=0)
#
#     # M+1-sequences in range if M-sequences are in range & last values are close
#     Dp = np.abs(
#         np.tile(X[M:], (N - M, 1)) - np.tile(X[M:], (N - M, 1)).T
#     )
#
#     Cmp = np.logical_and(Dp <= R, InRange[:-1, :-1]).mean(axis=0)
#
#     Phi_m, Phi_mp = np.sum(np.log(Cm)), numpy.sum(np.log(Cmp))
#
#     Ap_En = (Phi_m - Phi_mp) / (N - M)
#
#     return Ap_En
#
#
# def spectral_entropy(X, Band=[0.5,4,8,13,30,100], Fs=125, Power_Ratio=None):
#     """Compute spectral entropy of a time series from either two cases below:
#     1. X, the time series (default)
#     2. Power_Ratio, a list of normalized signal power in a set of frequency
#     bins defined in Band (if Power_Ratio is provided, recommended to speed up)
#
#     In case 1, Power_Ratio is computed by bin_power() function.
#
#     Notes
#     -----
#     To speed up, it is recommended to compute Power_Ratio before calling this
#     function because it may also be used by other functions whereas computing
#     it here again will slow down.
#
#     Parameters
#     ----------
#
#     Band
#         list
#
#         boundary frequencies (in Hz) of bins. They can be unequal bins, e.g.
#         [0.5,4,7,12,30] which are delta, theta, alpha and beta respectively.
#         You can also use range() function of Python to generate equal bins and
#         pass the generated list to this function.
#
#         Each element of Band is a physical frequency and shall not exceed the
#         Nyquist frequency, i.e., half of sampling frequency.
#
#      X
#         list
#
#         a 1-D real time series.
#
#     Fs
#         integer
#
#         the sampling rate in physical frequency
#
#     Returns
#     -------
#
#     As indicated in return line
#
#     See Also
#     --------
#     bin_power: pyeeg function that computes spectral power in frequency bins
#
#     """
#
#     if Power_Ratio is None:
#         Power, Power_Ratio = bin_power(X, Band, Fs)
#
#     Spectral_Entropy = 0
#     for i in range(0, len(Power_Ratio) - 1):
#         Spectral_Entropy += Power_Ratio[i] * numpy.log(Power_Ratio[i])
#     Spectral_Entropy /= numpy.log(
#         len(Power_Ratio)
#     )  # to save time, minus one is omitted
#     return -1 * Spectral_Entropy


# def embed_seq(time_series, tau, embedding_dimension):
#     """Build a set of embedding sequences from given time series `time_series`
#     with lag `tau` and embedding dimension `embedding_dimension`.
#
#     Let time_series = [x(1), x(2), ... , x(N)], then for each i such that
#     1 < i <  N - (embedding_dimension - 1) * tau,
#     we build an embedding sequence,
#     Y(i) = [x(i), x(i + tau), ... , x(i + (embedding_dimension - 1) * tau)].
#
#     All embedding sequences are placed in a matrix Y.
#
#     Parameters
#     ----------
#
#     time_series
#         list or numpy.ndarray
#
#         a time series
#
#     tau
#         integer
#
#         the lag or delay when building embedding sequence
#
#     embedding_dimension
#         integer
#
#         the embedding dimension
#
#     Returns
#     -------
#
#     Y
#         2-embedding_dimension list
#
#         embedding matrix built
#
#     Examples
#     ---------------
#     >>> import pyeeg
#     >>> a=range(0,9)
#     >>> pyeeg.embed_seq(a,1,4)
#     array([[0,  1,  2,  3],
#            [1,  2,  3,  4],
#            [2,  3,  4,  5],
#            [3,  4,  5,  6],
#            [4,  5,  6,  7],
#            [5,  6,  7,  8]])
#     >>> pyeeg.embed_seq(a,2,3)
#     array([[0,  2,  4],
#            [1,  3,  5],
#            [2,  4,  6],
#            [3,  5,  7],
#            [4,  6,  8]])
#     >>> pyeeg.embed_seq(a,4,1)
#     array([[0],
#            [1],
#            [2],
#            [3],
#            [4],
#            [5],
#            [6],
#            [7],
#            [8]])
#
#     """
#     if not type(time_series) == numpy.ndarray:
#         typed_time_series = numpy.asarray(time_series)
#     else:
#         typed_time_series = time_series
#
#     shape = (
#         typed_time_series.size - tau * (embedding_dimension - 1),
#         embedding_dimension
#     )
#
#     strides = (typed_time_series.itemsize, tau * typed_time_series.itemsize)
#
#     return numpy.lib.stride_tricks.as_strided(
#         typed_time_series,
#         shape=shape,
#         strides=strides
#     )