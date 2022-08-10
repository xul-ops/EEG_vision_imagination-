import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import os
from scipy.signal import welch
from scipy.integrate import simps
from mne.time_frequency import psd_array_multitaper
from scipy.fft import fft, fftfreq
from scipy.signal import periodogram
import pywt
import mne
from sklearn.preprocessing import MinMaxScaler

try:
    import pipelines.eeg_features as eeg_fe
except ModuleNotFoundError:
    import eeg_features as eeg_fe

try:
    import pipelines.EEGExtract as eeg_et
except ModuleNotFoundError:
    import eeg_features as eeg_fe



def plot_intervals(intervals_list, alphabet_list, asl_list, savefig_name="comparision.png", need_uV=True):
    """
    # item in intervals_list (0, (86634, 86994), 'imagination', 'asl', 9, 'I')

    """

    plot_names = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8', 'ch9', 'ch10',
                  'ch11', 'ch12', 'ch13', 'ch14', 'ch15', 'ch16']
    sample_rate = 125

    # find minest length
    lengths_list = list()
    for item in intervals_list:
        lengths_list.append(item[1][-1] - item[1][0])

    min_length = min(lengths_list) + 1
    # print(min_length)

    fig, ax = plt.subplots(16, figsize=(13, 25))
    k = 0
    time = np.arange(min_length) / sample_rate

    for name in plot_names:
        count = 0
        for item in intervals_list:

            if item[3] == 'asl':
                current_label = item[3] + '_' + item[2] + '_' + item[-1] + '_' + str(count)
                if need_uV:
                    ax[k].plot(time, asl_list[item[0]][name][item[1][-1] - 359:item[1][-1] + 1], label=current_label)
                else:
                    ax[k].plot(time, asl_list[item[0]][name][item[1][-1] - 359:item[1][-1] + 1] / 1000,
                               label=current_label)

            else:
                current_label = item[3] + '_' + item[2] + '_' + item[-1] + '_' + str(count)
                if need_uV:
                    ax[k].plot(time, alphabet_list[item[0]][name][item[1][-1] - 359:item[1][-1] + 1],
                               label=current_label)
                else:
                    ax[k].plot(time, alphabet_list[item[0]][name][item[1][-1] - 359:item[1][-1] + 1] / 1000,
                               label=current_label)
            count += 1

        if need_uV:
            ax[k].set_ylabel(name + '-uV')
        else:
            ax[k].set_ylabel(name + '-V')
        ax[k].set_xlabel('Time\(s)')
        ax[k].grid()
        ax[k].legend()
        k += 1

    plt.title('Comparision asl/alphabet imagination/vision ', y=19)
    # plt.show()
    fig.savefig(savefig_name)


def one_signal_band_power(data, sf=125, method='welch', window_sec=None, relative=False):
    """
    Compute the average power of the signal x in a specific frequency band. Requires MNE-Python >= 0.14.
    :param data:  (1d-array)  Input signal in the time-domain.
    :param sf:   Sampling frequency of the data.
    :param method: Periodogram method: 'welch' or 'multitaper', 'fft'
    :param window_sec: Length of each window in seconds. Useful only if method == 'welch'.
                        If None, window_sec = (1 / min(band)) * 2
    :param relative:    If True, return the relative power (= divided by the total power of the signal).
                        If False (default), return the absolute power.
    :return:  Absolute or relative band power.
    """

    low_delta, high_delta = 0.5, 4  # (0.1,4) (0.3, 4)
    low_theta, high_theta = 4, 8
    low_alpha, high_alpha = 8, 13
    low_beta, high_beta = 13, 30  # 13,32
    low_gamma, high_gamma = 30, 100  # 32, 100

    # Compute the modified periodogram (Welch)
    if method == 'welch':
        if window_sec is not None:
            nperseg = window_sec * sf
            freqs, psd = welch(data, sf, nperseg=nperseg)
        else:
            nperseg_delta = (2 / low_delta) * sf
            nperseg_theta = (2 / low_theta) * sf
            nperseg_alpha = (2 / low_alpha) * sf
            nperseg_beta = (2 / low_beta) * sf
            nperseg_gamma = (2 / low_delta) * sf

            freqs_delta, psd_delta = welch(data, sf, nperseg=nperseg_delta)
            freqs_theta, psd_theta = welch(data, sf, nperseg=nperseg_theta)
            freqs_alpha, psd_alpha = welch(data, sf, nperseg=nperseg_alpha)
            freqs_beta, psd_beta = welch(data, sf, nperseg=nperseg_beta)
            freqs_gamma, psd_gamma = welch(data, sf, nperseg=nperseg_gamma)

    elif method == 'multitaper':
        psd, freqs = psd_array_multitaper(data, sf, adaptive=True,
                                          normalization='full', verbose=0)

    elif method == 'fft':
        freqs, psd = periodogram(data, sf)

    if method == 'welch' and window_sec is None:
        # Frequency resolution
        freq_res_delta = freqs_delta[1] - freqs_delta[0]
        freq_res_theta = freqs_theta[1] - freqs_theta[0]
        freq_res_alpha = freqs_alpha[1] - freqs_alpha[0]
        freq_res_beta = freqs_beta[1] - freqs_beta[0]
        freq_res_gamma = freqs_gamma[1] - freqs_gamma[0]

        # Find index of band in frequency vector
        idx_delta = np.logical_and(freqs_delta >= low_delta, freqs_delta < high_delta)
        idx_theta = np.logical_and(freqs_theta >= low_theta, freqs_theta < high_theta)
        idx_alpha = np.logical_and(freqs_alpha >= low_alpha, freqs_alpha < high_alpha)
        idx_beta = np.logical_and(freqs_beta >= low_beta, freqs_beta < high_beta)
        idx_gamma = np.logical_and(freqs_gamma >= low_gamma, freqs_gamma <= high_gamma)

        # Integral approximation of the spectrum using parabola (Simpson's rule)
        delta = simps(psd_delta[idx_delta], dx=freq_res_delta)
        theta = simps(psd_theta[idx_theta], dx=freq_res_theta)
        alpha = simps(psd_alpha[idx_alpha], dx=freq_res_alpha)
        beta = simps(psd_beta[idx_beta], dx=freq_res_beta)
        gamma = simps(psd_gamma[idx_gamma], dx=freq_res_gamma)

        if relative:
            delta /= simps(psd_delta, dx=freq_res_delta)
            theta /= simps(psd_theta, dx=freq_res_theta)
            alpha /= simps(psd_alpha, dx=freq_res_alpha)
            beta /= simps(psd_beta, dx=freq_res_beta)
            gamma /= simps(psd_gamma, dx=freq_res_gamma)

        return [delta, theta, alpha, beta, gamma]

    else:

        # Frequency resolution
        freq_res = freqs[1] - freqs[0]
        # print(len(freq_res))

        # Find index of band in frequency vector
        idx_delta = np.logical_and(freqs >= low_delta, freqs <= high_delta)
        idx_theta = np.logical_and(freqs >= low_theta, freqs <= high_theta)
        idx_alpha = np.logical_and(freqs >= low_alpha, freqs <= high_alpha)
        idx_beta = np.logical_and(freqs >= low_beta, freqs <= high_beta)
        idx_gamma = np.logical_and(freqs >= low_gamma, freqs <= high_gamma)

        # Integral approximation of the spectrum using parabola (Simpson's rule)
        delta = simps(psd[idx_delta], dx=freq_res)
        theta = simps(psd[idx_theta], dx=freq_res)
        alpha = simps(psd[idx_alpha], dx=freq_res)
        beta = simps(psd[idx_beta], dx=freq_res)
        gamma = simps(psd[idx_gamma], dx=freq_res)

        if relative:
            delta /= simps(psd, dx=freq_res)
            theta /= simps(psd, dx=freq_res)
            alpha /= simps(psd, dx=freq_res)
            beta /= simps(psd, dx=freq_res)
            gamma /= simps(psd, dx=freq_res)

        return [delta, theta, alpha, beta, gamma]


def power_band(data, sf=125, method='welch', window_sec=None, relative=False):
    # data : 2-d array
    result = list()
    for i in range(data.shape[-1]):
        ch = data[:, i]
        current = one_signal_band_power(ch, sf=sf, method=method, window_sec=window_sec, relative=relative)
        result.append(current)

    return result


def power_band_timeslice(data, time_step=4, sf=125, method='welch', window_sec=None, relative=False):
    # data : 2-d array
    result = list()
    for i in range(data.shape[-1]):

        ch = data[:, i]
        each_slice_length = len(ch) // time_step
        bp = list()
        for j in range(time_step):

            if j != time_step - 1:
                current = ch[j * each_slice_length:each_slice_length * (j + 1)]
                current = one_signal_band_power(current, sf=sf, method=method, window_sec=window_sec, relative=relative)

            else:
                current = ch[j * each_slice_length:]
                current = one_signal_band_power(current, sf=sf, method=method, window_sec=window_sec, relative=relative)
            bp.append(current)
        result.append(bp)

    return result


def dwt_wavelet(signal, level=8, norm=True):
    """
    Compute discrete wavelet transform using pywt.
    Args:
    -----
            signal (dict): raw EEG signal from all channels
            level (int): level of DWT decomposition, default=7 bc of 200Hz sampling frequency
            norm (boolean): normalization by variation in signal, default=True

    Returns:
    --------
            data (array): arrays of coefficients for delta, theta, alpha and beta subbands
    """
    dwt_alpha = []
    dwt_theta = []
    dwt_delta = []
    dwt_beta = []

    for ch in signal:
        dwt = pywt.wavedec(ch, 'db4', mode='smooth', level=level)

        # dwt[0] is approx coeffs, and then starting at [1] the lowest decomposition lvl and ascending
        # so dwt[1] means delta subband

        if (norm):
            dwt_beta.append(np.mean(dwt[4] ** 2) / np.var(signal))
            dwt_alpha.append(np.mean(dwt[3] ** 2) / np.var(signal))
            dwt_theta.append(np.mean(dwt[2] ** 2) / np.var(signal))
            dwt_delta.append(np.mean(dwt[1] ** 2) / np.var(signal))
        else:
            dwt_beta.append(np.mean(dwt[4] ** 2))
            dwt_alpha.append(np.mean(dwt[3] ** 2))
            dwt_theta.append(np.mean(dwt[2] ** 2))
            dwt_delta.append(np.mean(dwt[1] ** 2))

    return [dwt_delta, dwt_theta, dwt_alpha, dwt_beta]


# iter_freqs = [
#     {'name': 'delta', 'fmin': 0.5, 'fmax': 4},
#     {'name': 'theta', 'fmin': 4, 'fmax': 8},
#     {'name': 'alpha', 'fmin': 8, 'fmax': 13},
#     {'name': 'beta', 'fmin': 13, 'fmax': 30},
#     {'name': 'gamma', 'fmin': 30, 'fmax': 100},
# ]
def TimeFrequencyWP(data, fs, wavelet, maxlevel=8):
    iter_freqs = [
        {'name': 'delta', 'fmin': 0.5, 'fmax': 4},
        {'name': 'theta', 'fmin': 4, 'fmax': 8},
        {'name': 'alpha', 'fmin': 8, 'fmax': 13},
        {'name': 'beta', 'fmin': 13, 'fmax': 30},
        {'name': 'gamma', 'fmin': 30, 'fmax': 100},
    ]
    results = list()
    wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)

    freqTree = [node.path for node in wp.get_level(maxlevel, 'freq')]
    # Calculate the bandwidth of the minimum frequency band of maxlevel
    freqBand = fs / (2 ** maxlevel)

    for iter in range(len(iter_freqs)):
        # Construct empty wavelet packet
        new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
        for i in range(len(freqTree)):
            # Minimum frequency of the ith band
            bandMin = i * freqBand
            # the maximum frequency of the ith band
            bandMax = bandMin + freqBand
            # Determine whether the i-th frequency band is within the range to be analyzed
            if (iter_freqs[iter]['fmin'] <= bandMin and iter_freqs[iter]['fmax'] >= bandMax):
                # Assign values to newly constructed wavelet packet parameters
                new_wp[freqTree[i]] = wp[freqTree[i]].data
        # data corresponding to frequency
        results.append(new_wp.reconstruct(update=True))

    return results


def eeg_feature(data, sf=125):
    # data : 2-d array
    result = dict()
    mm = MinMaxScaler()
    mm_data = mm.fit_transform(data)
    for i in range(data.shape[-1]):
        ch = data[:, i]
        mm_ch = mm_data[:, i]

        result['ch' + str(i + 1) + '_' + 'min'] = eeg_fe.min_X(ch)
        result['ch' + str(i + 1) + '_' + 'max'] = eeg_fe.max_X(ch)
        result['ch' + str(i + 1) + '_' + 'std'] = eeg_fe.std_X(ch)
        result['ch' + str(i + 1) + '_' + 'mean'] = eeg_fe.mean_X(ch)
        result['ch' + str(i + 1) + '_' + 'coefficient_variation'] = eeg_fe.coefficient_variation(ch)
        result['ch' + str(i + 1) + '_' + 'mean_abs'] = eeg_fe.mean_absolute_X(ch)
        result['ch' + str(i + 1) + '_' + 'AAC'] = eeg_fe.average_amplitude_change(ch)
        result['ch' + str(i + 1) + '_' + 'CARD'] = eeg_fe.cardinality(ch)
        # print(eeg_fe.carinality(ch))

        result['ch' + str(i + 1) + '_' + 'EMAV'] = eeg_fe.enhanced_mean_absolute(mm_ch)
        result['ch' + str(i + 1) + '_' + 'median'] = eeg_fe.median_X(ch)
        result['ch' + str(i + 1) + '_' + 'MAP'] = eeg_fe.mean_amplitude_power(ch)
        result['ch' + str(i + 1) + '_' + 'signal_energy'] = eeg_fe.signal_energy(ch)
        result['ch' + str(i + 1) + '_' + 'mean_energy'] = eeg_fe.mean_energy(ch)
        result['ch' + str(i + 1) + '_' + 'waveform_length'] = eeg_fe.waveform_length(ch)

        # all nan, pass this
        #         result['ch' + str(i+1) + '_' + 'EML'] =  eeg_fe.enhanced_wave_length(ch)
        #         print(eeg_fe.enhanced_wave_length(mm_ch))

        # sum_diff, min_diff, max_diff, mean_diff, median_diff
        diff_1 = eeg_fe.first_order_diff(ch)
        diff_2 = eeg_fe.second_order_diff(ch)
        result['ch' + str(i + 1) + '_' + '1_sum_diff'] = diff_1[0]
        result['ch' + str(i + 1) + '_' + '1_min_diff'] = diff_1[1]
        result['ch' + str(i + 1) + '_' + '1_max_diff'] = diff_1[2]
        result['ch' + str(i + 1) + '_' + '1_mean_diff'] = diff_1[3]
        result['ch' + str(i + 1) + '_' + '1_median_diff'] = diff_1[4]
        result['ch' + str(i + 1) + '_' + '2_sum_diff'] = diff_2[0]
        result['ch' + str(i + 1) + '_' + '2_min_diff'] = diff_2[1]
        result['ch' + str(i + 1) + '_' + '2_max_diff'] = diff_2[2]
        result['ch' + str(i + 1) + '_' + '2_mean_diff'] = diff_2[3]
        result['ch' + str(i + 1) + '_' + '2_median_diff'] = diff_2[4]

        result['ch' + str(i + 1) + '_' + 'log_energy_entropy'] = eeg_fe.log_energy_entropy(ch)
        # very close values
        result['ch' + str(i + 1) + '_' + 'renyi_entropy'] = eeg_fe.renyi_entropy(ch)
        # print(eeg_fe.renyi_entropy(ch))
        result['ch' + str(i + 1) + '_' + 'LRSSV'] = eeg_fe.log_root_sum_of_sequential_Variation(ch)
        result['ch' + str(i + 1) + '_' + 'MCL'] = eeg_fe.mean_curve_length(ch)
        result['ch' + str(i + 1) + '_' + 'mean_target_energy'] = eeg_fe.mean_teager_energy(ch)
        result['ch' + str(i + 1) + '_' + 'var'] = eeg_fe.var_X(ch)
        result['ch' + str(i + 1) + '_' + 'totalVariation'] = eeg_fe.totalVariation(ch)
        result['ch' + str(i + 1) + '_' + 'skew'] = eeg_fe.skew_X(ch)
        result['ch' + str(i + 1) + '_' + 'kurtosis'] = eeg_fe.kurs_X(ch)
        result['ch' + str(i + 1) + '_' + 'rms'] = eeg_fe.rms_X(ch)
        result['ch' + str(i + 1) + '_' + 'peak'] = eeg_fe.peak_X(ch)
        # closing
        result['ch' + str(i + 1) + '_' + 'PAPR'] = eeg_fe.papr_X(ch)

        #         # pEA5, pED5, pED4, pED3, pED2, pED1  wavelet; but wavelet power is not a standard power, so skip this feature for now
        #         wavelet_power = eeg_fe.relativePower(ch)
        #         result['ch' + str(i+1) + '_' + 'wavelet_power51'] =  wavelet_power[0]
        #         result['ch' + str(i+1) + '_' + 'wavelet_power52'] =  wavelet_power[1]
        #         result['ch' + str(i+1) + '_' + 'wavelet_power4'] =  wavelet_power[2]
        #         result['ch' + str(i+1) + '_' + 'wavelet_power3'] =  wavelet_power[3]
        #         result['ch' + str(i+1) + '_' + 'wavelet_power2'] =  wavelet_power[4]
        #         result['ch' + str(i+1) + '_' + 'wavelet_power1'] =  wavelet_power[5]

        result['ch' + str(i + 1) + '_' + 'wavelet_entropy'] = eeg_fe.wavelet_entopy(ch)
        result['ch' + str(i + 1) + '_' + 'hurst'] = eeg_fe.Hurst(ch)
        # closing
        result['ch' + str(i + 1) + '_' + 'PFD'] = eeg_fe.Petrosian_FD(ch)

        result['ch' + str(i + 1) + '_' + 'sample_entropy'] = eeg_fe.sample_entropy(ch)

        # two different permutation_entropy, I use ant
        PE = eeg_fe.permutation_entropy(ch)
        # result['ch' + str(i+1) + '_' + 'pye_permutation_entropy'] =  PE[0]
        result['ch' + str(i + 1) + '_' + 'ant_permutation_entropy'] = PE[1]

        hjorth = eeg_fe.Hjorth(ch)
        result['ch' + str(i + 1) + '_' + 'hjorth_activity'] = hjorth[0]
        result['ch' + str(i + 1) + '_' + 'hjorth_mobility'] = hjorth[1]
        result['ch' + str(i + 1) + '_' + 'hjorth_complexity'] = hjorth[-1]

        result['ch' + str(i + 1) + '_' + 'KFD'] = eeg_fe.KFD(ch)

        # two different DFA, I use ant
        DFA = eeg_fe.DFA(ch)
        result['ch' + str(i + 1) + '_' + 'DFA'] = DFA[0]
        #  result['ch' + str(i+1) + '_' + 'DFA_2'] = DFA[1]

        result['ch' + str(i + 1) + '_' + 'HFD'] = eeg_fe.HFD(ch)
        result['ch' + str(i + 1) + '_' + 'shannon_entropy'] = eeg_fe.shannon_entropy(ch)
        result['ch' + str(i + 1) + '_' + 'spectral_entropy'] = eeg_fe.spectral_entropy(ch)
        result['ch' + str(i + 1) + '_' + 'approximate_entropy'] = eeg_fe.approximate_entropy(ch)
        result['ch' + str(i + 1) + '_' + 'svd_entropy'] = eeg_fe.svd_entropy(ch)

        # all are 0
        result['ch' + str(i + 1) + '_' + 'num_zero_crossing'] = eeg_fe.num_zero_crossing(ch)
        # print(eeg_fe.num_zero_crossing(ch))

        # own LZC, not ant LZC, ant LZC is too slow
        result['ch' + str(i + 1) + '_' + 'LZC'] = eeg_fe.LZC(ch)
        # print(eeg_fe.LZC(ch))

        # welch power band
        pb = one_signal_band_power(ch, method='welch')
        result['ch' + str(i + 1) + '_' + 'pb_delta'] = pb[0]
        result['ch' + str(i + 1) + '_' + 'pb_theta'] = pb[1]
        result['ch' + str(i + 1) + '_' + 'pb_alpha'] = pb[2]
        result['ch' + str(i + 1) + '_' + 'pb_beta'] = pb[3]
        result['ch' + str(i + 1) + '_' + 'pb_gamma'] = pb[4]
        result['ch' + str(i + 1) + '_' + 'alpha/delta'] = pb[2] / pb[0]

    return result


def eeg_feature_2(data, fs=125):
    # eegData: 3D np array [chans x ms x epochs]
    eegData = np.array(data).reshape(16, -1, 1)
    result = dict()

    # # signals need to be normalized, otherwise sometimes there will be an error
    mm = MinMaxScaler()
    mm_data = mm.fit_transform(data)
    mm_eegData = np.array(mm_data).reshape(16, -1, 1)
    orders = [1]  # list(range(1,10+1))
    tsalisRes = eeg_et.tsalisEntropy(mm_eegData, bin_min=-200, bin_max=200, binWidth=2, orders=orders)
    tsalisRes = np.array(tsalisRes).reshape(16, )

    for i in range(16):
        result['ch' + str(i + 1) + '_' + 'tsalis_entropy'] = tsalisRes[i]

    # Subband Information Quantity
    eegData_delta = eeg_et.filt_data(eegData, 0.5, 4, fs)
    ShannonRes_delta = eeg_et.shannonEntropy(eegData_delta, bin_min=-200, bin_max=200, binWidth=2)
    eegData_theta = eeg_et.filt_data(eegData, 4, 8, fs)
    ShannonRes_theta = eeg_et.shannonEntropy(eegData_theta, bin_min=-200, bin_max=200, binWidth=2)
    eegData_alpha = eeg_et.filt_data(eegData, 8, 13, fs)
    ShannonRes_alpha = eeg_et.shannonEntropy(eegData_alpha, bin_min=-200, bin_max=200, binWidth=2)
    eegData_beta = eeg_et.filt_data(eegData, 13, 30, fs)
    ShannonRes_beta = eeg_et.shannonEntropy(eegData_beta, bin_min=-200, bin_max=200, binWidth=2)
    # fs should > 2*high, so we use 60
    eegData_gamma = eeg_et.filt_data(eegData, 30, 60, fs)
    ShannonRes_gamma = eeg_et.shannonEntropy(eegData_gamma, bin_min=-200, bin_max=200, binWidth=2)

    for i in range(16):
        result['ch' + str(i + 1) + '_' + 'PB_SE_1'] = ShannonRes_delta[i, 0]
        result['ch' + str(i + 1) + '_' + 'PB_SE_2'] = ShannonRes_theta[i, 0]
        result['ch' + str(i + 1) + '_' + 'PB_SE_3'] = ShannonRes_alpha[i, 0]
        result['ch' + str(i + 1) + '_' + 'PB_SE_4'] = ShannonRes_beta[i, 0]
        result['ch' + str(i + 1) + '_' + 'PB_SE_5'] = ShannonRes_gamma[i, 0]

    # Cepstrum Coefficients (n=2)
    CepstrumRes = eeg_et.mfcc(eegData, fs, order=2)
    CepstrumRes = np.array(CepstrumRes).reshape(16, 2)
    for i in range(16):
        result['ch' + str(i + 1) + '_' + 'cepstrum_1'] = CepstrumRes[i, 0]
        result['ch' + str(i + 1) + '_' + 'cepstrum_2'] = CepstrumRes[i, 1]

    # Lyapunov Exponent
    LyapunovRes = eeg_et.lyapunov(eegData)
    for i in range(16):
        result['ch' + str(i + 1) + '_' + 'lyapunov_exponent'] = LyapunovRes[i, 0]

    # Median Frequency
    medianFreqRes = eeg_et.medianFreq(eegData, fs)
    for i in range(16):
        result['ch' + str(i + 1) + '_' + 'median_frequency'] = medianFreqRes[i, 0]

    # # Mean Frequency
    # meanFreqRes = eeg_et.meanFreq(eegData, fs)
    # for i in range(16):
    #     result['ch' + str(i + 1) + '_' + 'mean_frequency'] = meanFreqRes[i, 0]

    # Regularity (burst-suppression)
    regularity_res = eeg_et.eegRegularity(eegData, fs)
    for i in range(16):
        result['ch' + str(i + 1) + '_' + 'regularity'] = regularity_res[i, 0]

    # below, a lot of features are same 0 in example

    # False Nearest Neighbor
    FalseNnRes = eeg_et.falseNearestNeighbor(eegData)
    for i in range(16):
        result['ch' + str(i + 1) + '_' + 'FNN'] = FalseNnRes[i, 0]

    # Diffuse Slowing
    df_res = eeg_et.diffuseSlowing(eegData)
    for i in range(16):
        result['ch' + str(i + 1) + '_' + 'diffuse_slowing'] = df_res[i, 0]

    # Spikes
    minNumSamples = int(70 * fs / 1000)
    spikeNum_res = eeg_et.spikeNum(eegData, minNumSamples)
    for i in range(16):
        result['ch' + str(i + 1) + '_' + 'spikes'] = spikeNum_res[i, 0]

    # # Delta burst after Spike
    # deltaBurst_res = eeg_et.burstAfterSpike(eegData,eegData_delta,minNumSamples=7,stdAway = 3)
    # deltaBurst_res

    # Sharp spike
    sharpSpike_res = eeg_et.shortSpikeNum(eegData, minNumSamples)
    for i in range(16):
        result['ch' + str(i + 1) + '_' + 'sharp_spikes'] = sharpSpike_res[i, 0]

    # Number of Bursts
    numBursts_res = eeg_et.numBursts(eegData, fs)
    for i in range(16):
        result['ch' + str(i + 1) + '_' + 'num_burst'] = numBursts_res[i, 0]

    # Burst length μ and σ
    burstLenMean_res, burstLenStd_res = eeg_et.burstLengthStats(eegData, fs)
    for i in range(16):
        result['ch' + str(i + 1) + '_' + 'burst_length_mean'] = burstLenMean_res[i, 0]
        result['ch' + str(i + 1) + '_' + 'burst_length_std'] = burstLenStd_res[i, 0]

    # Number of Suppressions
    numSupps_res = eeg_et.numSuppressions(eegData, fs)
    for i in range(16):
        result['ch' + str(i + 1) + '_' + 'supressions'] = numSupps_res[i, 0]

    # Suppression length μ and σ
    suppLenMean_res, suppLenStd_res = eeg_et.suppressionLengthStats(eegData, fs)
    for i in range(16):
        result['ch' + str(i + 1) + '_' + 'supressions_length_mean'] = suppLenMean_res[i, 0]
        result['ch' + str(i + 1) + '_' + 'supressions_length_std'] = suppLenStd_res[i, 0]

    # all same 1
    # Connectivity features- Coherence - δ
    coherence_res = eeg_et.coherence(eegData, fs)
    for i in range(16):
        result['ch' + str(i + 1) + '_' + 'coherence'] = coherence_res[i, 0]

    return result


def generate_feature_dict(data):
    fe_1 = eeg_feature(np.array(data))
    fe_2 = eeg_feature_2(np.array(data))
    feature_dict = dict(fe_1, **fe_2)

    return feature_dict


