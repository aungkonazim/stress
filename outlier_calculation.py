import numpy as np
from scipy.stats import iqr
from enum import Enum


class Quality(Enum):
    ACCEPTABLE = 1
    UNACCEPTABLE = 0

def outlier_computation(valid_rr_interval_time: list,
                        valid_rr_interval_sample: list,
                        criterion_beat_difference: float):
    """
    This function implements the rr interval outlier calculation through comparison with the criterion
    beat difference and consecutive differences with the previous and next sample

    :param valid_rr_interval_time: A python array of rr interval time
    :param valid_rr_interval_sample: A python array of rr interval samples
    :param criterion_beat_difference: A threshold calculated from the RR interval data passed

    yields: The quality of each data point in the RR interval array
    """
    standard_rr_interval_sample = valid_rr_interval_sample[0]
    previous_rr_interval_quality = Quality.ACCEPTABLE

    for i in range(1, len(valid_rr_interval_sample) - 1):

        rr_interval_diff_with_last_good = abs(standard_rr_interval_sample - valid_rr_interval_sample[i])
        rr_interval_diff_with_prev_sample = abs(valid_rr_interval_sample[i - 1] - valid_rr_interval_sample[i])
        rr_interval_diff_with_next_sample = abs(valid_rr_interval_sample[i] - valid_rr_interval_sample[i + 1])

        if previous_rr_interval_quality == Quality.UNACCEPTABLE and rr_interval_diff_with_last_good < criterion_beat_difference:
            yield (valid_rr_interval_time[i], Quality.ACCEPTABLE)
            previous_rr_interval_quality = Quality.ACCEPTABLE
            standard_rr_interval_sample = valid_rr_interval_sample[i]

        elif previous_rr_interval_quality == Quality.UNACCEPTABLE and rr_interval_diff_with_last_good > criterion_beat_difference >= rr_interval_diff_with_prev_sample and rr_interval_diff_with_next_sample <= criterion_beat_difference:
            yield (valid_rr_interval_time[i], Quality.ACCEPTABLE)
            previous_rr_interval_quality = Quality.ACCEPTABLE
            standard_rr_interval_sample = valid_rr_interval_sample[i]

        elif previous_rr_interval_quality == Quality.UNACCEPTABLE and rr_interval_diff_with_last_good > criterion_beat_difference and (
                        rr_interval_diff_with_prev_sample > criterion_beat_difference or rr_interval_diff_with_next_sample > criterion_beat_difference):
            yield (valid_rr_interval_time[i], Quality.UNACCEPTABLE)
            previous_rr_interval_quality = Quality.UNACCEPTABLE

        elif previous_rr_interval_quality == Quality.ACCEPTABLE and rr_interval_diff_with_prev_sample <= criterion_beat_difference:
            yield (valid_rr_interval_time[i], Quality.ACCEPTABLE)
            previous_rr_interval_quality = Quality.ACCEPTABLE
            standard_rr_interval_sample = valid_rr_interval_sample[i]

        elif previous_rr_interval_quality == Quality.ACCEPTABLE and rr_interval_diff_with_prev_sample > criterion_beat_difference:
            yield (valid_rr_interval_time[i], Quality.UNACCEPTABLE)
            previous_rr_interval_quality = Quality.UNACCEPTABLE

        else:
            yield (valid_rr_interval_time[i], Quality.UNACCEPTABLE)


def compute_outlier_ecg(ecg_ts,ecg_rr):
    """
    Reference - Berntson, Gary G., et al. "An approach to artifact identification: Application to heart period data."
    Psychophysiology 27.5 (1990): 586-598.

    :param ecg_rr: RR interval datastream

    :return: An annotated datastream specifying when the ECG RR interval datastream is acceptable
    """


    valid_rr_interval_sample = [i for i in ecg_rr if i > .3 and i < 2]
    valid_rr_interval_time = [ecg_ts[i] for i in range(len(ecg_ts)) if ecg_rr[i] > .3 and ecg_rr[i] < 2]
    valid_rr_interval_difference = abs(np.diff(valid_rr_interval_sample))

    # Maximum Expected Difference(MED)= 3.32* Quartile Deviation
    maximum_expected_difference = 4.5 * 0.5 * iqr(valid_rr_interval_difference)

    # Shortest Expected Beat(SEB) = Median Beat – 2.9 * Quartile Deviation
    # Minimal Artifact Difference(MAD) = SEB/ 3
    maximum_artifact_difference = (np.median(valid_rr_interval_sample) - 2.9 * .5 * iqr(
        valid_rr_interval_difference)) / 3

    # Midway between MED and MAD is considered
    criterion_beat_difference = (maximum_expected_difference + maximum_artifact_difference) / 2
    if criterion_beat_difference < .2:
        criterion_beat_difference = .2

    ecg_rr_quality_array = [(valid_rr_interval_time[0], Quality.ACCEPTABLE)]

    for data in outlier_computation(valid_rr_interval_time, valid_rr_interval_sample, criterion_beat_difference):
        ecg_rr_quality_array.append(data)
    ecg_rr_quality_array.append((valid_rr_interval_time[-1], Quality.ACCEPTABLE))
    return ecg_rr_quality_array
