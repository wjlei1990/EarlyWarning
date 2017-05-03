"""
Created on Wed Apr 26 14:36:27 2017

@author: Yajun Peng, Wenjie Lei
"""
from obspy.signal.trigger import trigger_onset
import matplotlib.pyplot as plt
from obspy.signal.trigger import plot_trigger, recursive_sta_lta
from proc_utils import highpass_filter_waveform


plt.style.use('seaborn-darkgrid')


def extract_arrival_index(arrival):
    if arrival["P_pick"] is not None:
        return arrival["P_pick"]["index"], "P_pick"
    elif arrival["P_pick_hfreq"] is not None:
        return arrival["P_pick_hfreq"]["index"], "P_pick_hfreq"
    elif arrival["P_pick_SNR"] is not None:
        return arrival["P_pick_SNR"]["index"], "P_pick_SNR"
    else:
        return None, None


def plot_stream_and_arrival(st, picks, origin_time, predict_arrivals):
    print("--------> Plot...")
    print(origin_time)
    plt.figure(figsize=(10, 15))
    ntraces = len(st)
    for idx, tr in enumerate(st):
        ax = plt.subplot(ntraces, 1, idx+1)
        ax.axis('off')
        # ax.get_xaxis().set_visible(False)
        plt.plot(tr.data)
        arrival_index, pick_type = extract_arrival_index(picks[tr.id])
        ymin, ymax = ax.get_ylim()
        event_index = (origin_time - tr.stats.starttime) / tr.stats.delta
        plt.vlines([event_index], ymin, ymax, linestyles="dotted",
                   color='g', linewidth=2)
        pred_arrival_index = \
            event_index + predict_arrivals[idx] / tr.stats.delta
        plt.vlines([pred_arrival_index], ymin, ymax, linestyles="dashdot",
                   color='k', linewidth=3)
        if pick_type is not None:
            plt.vlines([arrival_index], ymin, ymax, color='r', linewidth=2)

    plt.tight_layout()
    plt.show()


def check_arrival_time(arrivals, waveform_start_time,
                       origin_time, df):
    """
    Check if the arrival time is after the origin time.
    """
    for i_arrival in range(len(arrivals)):
        time_pick = waveform_start_time + float(arrivals[i_arrival][0] / df)
        if time_pick > origin_time:
            return {"time": time_pick, "index": arrivals[i_arrival][0]}

    return None


def pick_arrival(trace, nsta_seconds, nlta_seconds, df,
                 origin_time, pick_threshold,
                 plot_flag=False):
    """
    P wave arrival is picked using a recursive sta/lta algorithm.
    """
    cft = recursive_sta_lta(
        trace, int(nsta_seconds * df), int(nlta_seconds * df))

    arrivals = trigger_onset(cft, pick_threshold, 0.5)

    if plot_flag:
        plot_trigger(trace, cft, pick_threshold, 0.5, show=True)

    P_pick = check_arrival_time(
        arrivals, trace.stats.starttime, origin_time, df)

    return P_pick


def p_wave_onset_and_SNR(trace, origin_time, pre_filter=False,
                         pre_filter_freq=0.75,
                         pick_filter=True, pick_filter_freq=1.0,
                         tigger_threshold=100, SNR_threshold=100,
                         nsta_seconds=0.05, nlta_seconds=20,
                         SNR_plot_flag=False, trigger_plot_flag=False):
    """
    Pick P wave (seimic wave that arrives first) onset time
    from the vertical component using a recursive STA/LTA
    method (Withers et al., 1998). Signal-to-noise ratio
    is determined from the STA/LTA characteristic function.

    :param trace: obspy trace containing one seismogram.
    :param origin_time: earthquake origin time in UTCDateTime format.
    :param pre_filter: set True to first apply a highpass filter
        (> 0.075 Hz) to the stream.
    :param pick_filter: set True to apply a highpass filter (> 1 Hz
        to pick the onset. This should improve the onset accuracy.
    :Return:P_onset, dict:
        {station name: (
            (arrival in UTCDateTime, arrival in samples since start_time),
            boolean indicating whether SNR > SNR_threshold)}
    """
    # Check SNR using P wave arrival pick for the unfiltered waveform
    P_pick_SNR = None
    # P wave arrival pick for highpass filtered (> 1 Hz) waveform
    P_pick_hfreq = None

    if pre_filter:
        highpass_filter_waveform(trace, pre_filter_freq)

    # The main objective is to calculate SNR. But can be used to pick
    # arrival time when pick_filter = False.
    df = trace.stats.sampling_rate
    P_pick_SNR = pick_arrival(trace, nsta_seconds, nlta_seconds, df,
                              origin_time, SNR_threshold,
                              plot_flag=SNR_plot_flag)

    if pick_filter:
        trace_copy = trace.copy()
        highpass_filter_waveform(trace_copy, pick_filter_freq)

        # Pick arrival again using highpassed waveform
        P_pick_hfreq = pick_arrival(trace_copy, nsta_seconds,
                                    nlta_seconds, df,
                                    origin_time, tigger_threshold,
                                    plot_flag=trigger_plot_flag)

    # Check consistency between P_pick_hfreq and P_pick_SNR
    # Use P_pick_hfreq when the two values are within 0.5 s.
    P_pick = None
    if P_pick_hfreq and P_pick_SNR:
        if len(P_pick_SNR) and len(P_pick_hfreq):
            if abs(P_pick_hfreq["time"] - P_pick_SNR["time"]) < 0.5:
                P_pick = P_pick_hfreq
            else:
                P_pick = None

    return {"P_pick": P_pick, "P_pick_SNR": P_pick_SNR,
            "P_pick_hfreq": P_pick_hfreq}
