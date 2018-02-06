# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 19:18:28 2018

@author: aungkon
"""
from typing import List
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
from scipy import signal
import pandas as pd

def rr_interval_update(rpeak_temp1: List,
                       rr_ave: float,
                       min_size: int = 8) -> float:
    """
   :param min_size: 8 last R-peaks are checked to compute the running rr interval average
   :param rpeak_temp1: R peak locations
   :param rr_ave: previous rr-interval average
   :return: the new rr-interval average of the previously detected 8 R peak locations
   """
    peak_interval = np.diff([0] + rpeak_temp1)  # TODO: rpeak_temp1 is a datapoint, what should this be converted to?
    return rr_ave if len(peak_interval) < min_size else np.sum(peak_interval[-min_size:]) / min_size


def compute_moving_window_int(sample: np.ndarray,
                              fs: float,
                              blackman_win_length: int,
                              filter_length: int = 257,
                              delta: float = .02) -> np.ndarray:
    """
    :param sample: ecg sample array
    :param fs: sampling frequency
    :param blackman_win_length: length of the blackman window on which to compute the moving window integration
    :param filter_length: length of the FIR bandpass filter on which filtering is done on ecg sample array
    :param delta: to compute the weights of each band in FIR filter
    :return: the Moving window integration of the sample array
    """
    # I believe these constants can be kept in a file

    # filter edges
    filter_edges = [0, 4.5 * 2 / fs, 5 * 2 / fs, 20 * 2 / fs, 20.5 * 2 / fs, 1]
    # gains at filter band edges
    gains = [0, 0, 1, 1, 0, 0]
    # weights
    weights = [500 / delta, 1 / delta, 500 / delta]
    # length of the FIR filter

    # FIR filter coefficients for bandpass filtering
    filter_coeff = signal.firls(filter_length, filter_edges, gains, weights)

    # bandpass filtered signal
    bandpass_signal = signal.convolve(sample, filter_coeff, 'same')
    bandpass_signal /= np.percentile(bandpass_signal, 90)

    # derivative array
    derivative_array = (np.array([-1.0, -2.0, 0, 2.0, 1.0])) * (1 / 8)
    # derivative signal (differentiation of the bandpass)
    derivative_signal = signal.convolve(bandpass_signal, derivative_array, 'same')
    derivative_signal /= np.percentile(derivative_signal, 90)

    # squared derivative signal
    derivative_squared_signal = derivative_signal ** 2
    derivative_squared_signal /= np.percentile(derivative_squared_signal, 90)

    # blackman window
    blackman_window = np.blackman(blackman_win_length)
    # moving window Integration of squared derivative signal
    mov_win_int_signal = signal.convolve(derivative_squared_signal, blackman_window, 'same')
    mov_win_int_signal /= np.percentile(mov_win_int_signal, 90)

    return mov_win_int_signal


def check_peak(data: List) -> bool:
    """
    This is a function to check the condition of a simple peak of signal y in index i
    :param data:
    :return:
    """

    if len(data) < 3:
        return False

    midpoint = int(len(data) / 2)
    test_value = data[0]

    for i in data[1:midpoint + 1]:
        if test_value < i:
            test_value = i
        else:
            return False

    for i in data[midpoint + 1:]:
        if test_value > i:
            test_value = i
        else:
            return False

    return True


# TODO: CODE_REVIEW: Justify in the method documentation string the justification of the default values
# TODO: CODE_REVIEW: Make hard-coded constants default method parameter
def compute_r_peaks(threshold_1: float,
                    rr_ave: float,
                    mov_win_int_signal: np.ndarray,
                    peak_tuple_array: List[tuple]) -> list:
    """
    This function does the adaptive thresholding of the signal to get the R-peak locations


    :param threshold_1: Thr1 is the threshold above which the R peak
    :param rr_ave: running RR-interval average
    :param mov_win_int_signal: signal sample array
    :param peak_tuple_array: A tuple array containing location and values of the simple peaks detected in the process before

    :returns rpeak_array_indices: The location of the R peaks in the signal sample array once found this is returned


    """

    peak_location_in_signal_array = [i[0] for i in peak_tuple_array]  # location of the simple peaks in signal array
    amplitude_in_peak_locations = [i[1] for i in peak_tuple_array]  # simple peak's amplitude in signal array

    threshold_2 = 0.5 * threshold_1  # any signal value between threshold_2 and threshold_1 is a noise peak
    sig_lev = 4 * threshold_1  # current signal level -any signal above thrice the signal level is discarded as a spurious value
    noise_lev = 0.1 * sig_lev  # current noise level of the signal
    ind_rpeak = 0
    rpeak_array_indices = []
    rpeak_inds_in_peak_array = []
    while ind_rpeak < len(peak_location_in_signal_array):

        # if for 166 percent of the present RR interval no peak is detected as R peak then threshold_2 is taken as the
        # R peak threshold and the maximum of the range is taken as a R peak and RR interval is updated accordingly
        if len(rpeak_array_indices) >= 1 and peak_location_in_signal_array[ind_rpeak] - peak_location_in_signal_array[
            rpeak_inds_in_peak_array[-1]] > 1.66 * rr_ave and ind_rpeak - rpeak_inds_in_peak_array[-1] > 1:

            # values and indexes of previous peaks discarded as not an R peak whose magnitude is above threshold_2
            searchback_array = [(k - rpeak_inds_in_peak_array[-1], amplitude_in_peak_locations[k]) for k in
                                range(rpeak_inds_in_peak_array[-1] + 1, ind_rpeak) if
                                3 * sig_lev > amplitude_in_peak_locations[k] > threshold_2]

            if len(searchback_array) > 0:
                # maximum inside the range calculated beforehand is taken as R peak
                searchback_array_inrange_values = [x[1] for x in searchback_array]
                searchback_max_index = np.argmax(searchback_array_inrange_values)
                rpeak_array_indices.append(peak_location_in_signal_array[
                                               rpeak_inds_in_peak_array[-1] + searchback_array[searchback_max_index][
                                                   0]])
                rpeak_inds_in_peak_array.append(
                    rpeak_inds_in_peak_array[-1] + searchback_array[searchback_max_index][0])
                sig_lev = ewma(sig_lev, mov_win_int_signal[peak_location_in_signal_array[ind_rpeak]],
                               .125)  # update the current signal level
                threshold_1 = noise_lev + 0.25 * (sig_lev - noise_lev)
                threshold_2 = 0.5 * threshold_1
                rr_ave = rr_interval_update(rpeak_array_indices, rr_ave)
                ind_rpeak = rpeak_inds_in_peak_array[-1] + 1
            else:
                threshold_1 = noise_lev + 0.25 * (sig_lev - noise_lev)
                threshold_2 = 0.5 * threshold_1
                ind_rpeak += 1
        else:
            # R peak checking
            if threshold_1 <= mov_win_int_signal[peak_location_in_signal_array[ind_rpeak]] < 3 * sig_lev:
                rpeak_array_indices.append(peak_location_in_signal_array[ind_rpeak])
                rpeak_inds_in_peak_array.append(ind_rpeak)
                sig_lev = ewma(sig_lev, mov_win_int_signal[peak_location_in_signal_array[ind_rpeak]],
                               .125)  # update the signal level
            # noise peak checking
            elif threshold_1 > mov_win_int_signal[peak_location_in_signal_array[ind_rpeak]] > threshold_2:
                noise_lev = ewma(noise_lev, mov_win_int_signal[peak_location_in_signal_array[ind_rpeak]],
                                 .125)  # update the noise level
            threshold_1 = noise_lev + 0.25 * (sig_lev - noise_lev)
            threshold_2 = 0.5 * threshold_1
            ind_rpeak += 1
            rr_ave = rr_interval_update(rpeak_array_indices, rr_ave)
    return rpeak_array_indices


def ewma(value: float, new_value: float, alpha: float) -> float:
    """

    :param value:
    :param new_value:
    :param alpha:
    :return:
    """
    return alpha * new_value + (1 - alpha) * value


# TODO: CODE_REVIEW: Justify in the method documentation string the justification of the default values
# TODO: CODE_REVIEW: Make hard-coded constants default method parameter
def remove_close_peaks(rpeak_temp1: list,
                       sample: np.ndarray,
                       fs: float,
                       min_range: float = .4) -> list:
    """
    This function removes one of two peaks from two consecutive R peaks
    if difference among them is less than the minimum possible

    :param min_range:
    :param rpeak_temp1: R peak array containing the index of the R peaks
    :param sample: sample array
    :param fs: sampling frequency
    :return: R peak array with no close R peaks

    """
    sample = np.array(sample)
    difference = 0
    rpeak_temp2 = rpeak_temp1
    while difference != 1:
        length_rpeak_temp2 = len(rpeak_temp2)
        temp = np.diff(rpeak_temp2)
        comp_index1 = np.array([rpeak_temp2[i] for i in range(len(temp)) if temp[i] < min_range * fs])
        comp_index2 = np.array([rpeak_temp2[i + 1] for i in range(len(temp)) if temp[i] < min_range * fs])
        if len(comp_index1) > 0 and len(comp_index2) > 0:
            comp1 = sample[comp_index1]
            comp2 = sample[comp_index2]
            checkmin = np.matrix([comp1, comp2])
            temp_ind1 = [i for i in range(len(temp)) if temp[i] < min_range * fs]
            temp_ind2 = np.argmin(np.array(checkmin), axis=0)
            temp_ind = temp_ind1 + temp_ind2
            temp_ind = np.unique(temp_ind)
            count = 0
            for i in temp_ind:
                rpeak_temp2.remove(rpeak_temp2[i - count])
                count = count + 1
            difference = length_rpeak_temp2 - len(rpeak_temp2) + 1
            # print(difference)
        else:
            break
    return rpeak_temp2


def confirm_peaks(rpeak_temp1: list,
                  sample: np.ndarray,
                  fs: float,
                  range_for_checking: float = 1 / 10) -> np.ndarray:
    """

    This function does the final check on the R peaks detected and
    finds the maximum in a range of fs/10 of the detected peak location and assigns it to be the peak

    :param rpeak_temp1: R peak array containing the index of the R peaks
    :param sample: sample array
    :param fs: sampling frequency
    :param range_for_checking : The peaks are checked within a range of fs/10 to get the maximum value within that range

    :return: final R peak array

    """
    for i in range(1, len(rpeak_temp1) - 1):
        start_index = int(rpeak_temp1[i] - np.ceil(range_for_checking * fs))
        end_index = int(rpeak_temp1[i] + np.ceil(range_for_checking * fs) + 1)

        index = np.argmax(sample[start_index:end_index])

        rpeak_temp1[i] = rpeak_temp1[i] - np.ceil(range_for_checking * fs) + index

    return np.array(rpeak_temp1).astype(np.int64)


# TODO: CODE_REVIEW: Make hard-coded constants default method parameter
def detect_rpeak(ecg: list,
                 fs: float = 25,
                 threshold: float = 0.5,
                 blackman_win_len_range: float = 0.2) -> list:
    """
    This program implements the Pan Tomkins algorithm on ECG signal to detect the R peaks

    Since the ecg array can have discontinuity in the timestamp arrays the rr-interval calculated
    in the algorithm is calculated in terms of the index in the sample array

    The algorithm consists of some major steps

    1. computation of the moving window integration of the signal in terms of blackman window of a prescribed length
    2. compute all the peaks of the moving window integration signal
    3. adaptive thresholding with dynamic signal and noise thresholds applied to filter out the R peak locations
    4. confirm the R peaks through differentiation from the nearby peaks and remove the false peaks

    :param ecg: ecg array of tuples (timestamp,value)
    :param fs: sampling frequency
    :param threshold: initial threshold to detect the R peak in a signal normalized by the 90th percentile. .5 is default.
    :param blackman_win_len_range : the range to calculate blackman window length

    :return: R peak array of tuples (timestamp, Rpeak interval)
    """

    data = ecg
    result = []
    if len(data) == 0:
        result = []
        return result
    sample = data
    # computes the moving window integration of the signal
    blackman_win_len = np.ceil(fs * blackman_win_len_range)
    y = compute_moving_window_int(sample, fs, blackman_win_len)

    peak_location_values = [(i, y[i]) for i in range(2, len(y) - 1) if check_peak(y[i - 2:i + 3])]

    # initial RR interval average
    peak_location = [i[0] for i in peak_location_values]
    running_rr_avg = sum(np.diff(peak_location)) / (len(peak_location) - 1)

    rpeak_temp1 = compute_r_peaks(threshold, running_rr_avg, y, peak_location_values)
    # print(len(rpeak_temp1),len(sample),fs)
    rpeak_temp2 = remove_close_peaks(rpeak_temp1, sample, fs)
    index = confirm_peaks(rpeak_temp2, sample, fs)
    return np.array(index)





























