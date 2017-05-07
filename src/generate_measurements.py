"""
1) use the arrival time to generate windows
2) use the window and seismogram to generate measurements(inside windows)
"""
from __future__ import print_function, division
import os
import sys  # NOQA
import scipy.stats
import numpy as np
import pandas as pd
import obspy
from obspy import UTCDateTime
import matplotlib.pyplot as plt
from utils import load_json
from metric import cut_window, tau_c, tau_p_max, \
    make_disp_vel_acc_records
from metric2 import norm, envelope, split_accumul_data


def read_waveform(fn):
    st = obspy.read(fn, format="MSEED")
    return st


def measure_func(data, prefix):
    # plt.plot(data)
    # plt.xlabel(prefix)
    # plt.show()

    measure = {}
    npts = len(data)
    measure["%s.l1_norm" % prefix] = norm(data, ord=1) / npts
    measure["%s.abs_l1_norm" % prefix] = norm(np.abs(data), ord=1) / npts
    measure["%s.l2_norm" % prefix] = norm(data, ord=2) / npts
    measure["%s.l4_norm" % prefix] = norm(data, ord=4) / npts
    max_amp = np.max(np.abs(data))
    measure["%s.max_amp" % prefix] = max_amp
    max_amp_loc = np.argmax(np.abs(data)) / npts
    measure["%s.max_amp_loc" % prefix] = max_amp_loc
    measure["%s.max_amp_over_loc" % prefix] = max_amp / max_amp_loc

    _, _, mean, var, skew, kurt = scipy.stats.describe(np.abs(data))
    measure["%s.mean" % prefix] = mean
    measure["%s.var" % prefix] = var
    measure["%s.skew" % prefix] = skew
    measure["%s.kurt" % prefix] = kurt

    for perc in [25, 50, 75]:
        measure["%s.%d_perc" % (prefix, perc)] = \
            np.percentile(np.abs(data), perc)
    return measure


def measure_on_trace_data_type(trace, data_type, window_split):
    measure = {}

    channel = trace.stats.channel
    prefix = "%s.%s" % (channel, data_type)

    measure.update(measure_func(trace.data, prefix))

    data_split = split_accumul_data(trace.data, n=window_split)
    for idx in range(len(data_split)):
        prefix = "%s.%s.acumul_window_%d" % (channel, data_type, idx)
        measure.update(measure_func(data_split[idx], prefix))

    env_data = envelope(trace.data)
    prefix = "%s.%s.env" % (channel, data_type)
    measure.update(measure_func(env_data, prefix))

    env_split = split_accumul_data(env_data, n=window_split)
    for idx in range(len(data_split)):
        prefix = "%s.%s.env.window_%d" % (channel, data_type, idx)
        measure.update(measure_func(env_split[idx], prefix))

    return measure


def plot_arrival_window(trace, windows, origin_time):
    plt.plot(trace.data)
    ax = plt.gca()
    ymin, ymax = ax.get_ylim()

    idx = (UTCDateTime(windows["pick_arrival"]) - trace.stats.starttime) / \
        trace.stats.delta
    plt.vlines([idx], ymin, ymax, linestyles="dotted", color='r')

    idx = (UTCDateTime(windows["theo_arrival"]) - trace.stats.starttime) / \
        trace.stats.delta
    plt.vlines([idx], ymin, ymax, linestyles="dotted", color='b')

    idx = (origin_time - trace.stats.starttime) / \
        trace.stats.delta
    plt.vlines([idx], ymin, ymax, linestyles="dotted", color='b')

    plt.show()


def measure_tau_c(trace_types, window_split):
    channel = trace_types["disp"].stats.channel

    measure = {}
    measure["%s.tau_c" % channel] = \
        tau_c(trace_types["disp"].data, trace_types["vel"].data)

    disp_split = split_accumul_data(trace_types["disp"].data, n=window_split)
    vel_split = split_accumul_data(trace_types["vel"].data, n=window_split)
    for idx in range(len(disp_split)):
        measure["%s.tau_c.accumul_window_%d" % (channel, idx)] = \
            tau_c(disp_split[idx], vel_split[idx])

    return measure


def measure_on_trace(_trace, windows, src, win_len=3.0, window_split=6):
    arrival = UTCDateTime(windows["pick_arrival"])
    # plot_arrival_window(_trace, windows, UTCDateTime(src.time))

    trace = cut_window(_trace, arrival, win_len)
    print(trace)

    measure = {}
    _v = tau_p_max(trace)
    channel = trace.stats.channel
    measure["%s.tau_p_max" % channel] = _v["tau_p_max"]

    trace_types = make_disp_vel_acc_records(trace)
    measure.update(measure_tau_c(trace_types, window_split))

    for dtype, data in trace_types.iteritems():
        measure.update(measure_on_trace_data_type(data, dtype, window_split))

    # print(measure)
    # print("Number of features: %d" % len(measure))
    # print(measure.keys())
    return measure


def select_station_components(st, zchan):
    """
    Select the 3-component traces from the same channel
    """
    comps = ["Z", "N", "E"]
    st_select = obspy.Stream()
    for comp in comps:
        chan_id = zchan[:-1] + comp
        _st = st.select(id=chan_id)
        if len(_st) == 0:
            continue
        st_select.append(_st[0])

    return st_select


def measure_on_station_stream(src, st, chan_win):
    measure = {}
    for tr in st:
        _m = measure_on_trace(tr, chan_win, src)
        measure.update(_m)
        print("Number of measurements in trace: %d " % len(_m))

    # add common information
    measure["distance"] = chan_win["distance"]
    measure["channel"] = st[0].id
    measure["source"] = "%s" % UTCDateTime(src.time)
    measure["magnitude"] = src.mag
    return measure


def measure_on_stream(src, waveform_file, window_file):
    try:
        st = read_waveform(waveform_file)
    except Exception as err:
        print("Error reading waveform(%s): %s" % (waveform_file, err))
        return

    try:
        windows = load_json(window_file)
    except Exception as err:
        print("Error reading window file: %s" % err)
        return

    results = []
    missing_stations = 0
    for zchan, chan_win in windows.iteritems():
        st_comp = select_station_components(st, zchan)
        _nw = st_comp[0].stats.network
        _sta = st_comp[0].stats.station
        print("-" * 10 + " station: %s.%s " % (_nw, _sta) + "-" * 10)
        if len(st_comp) != 3:
            missing_stations += 1
            continue
        try:
            measure = measure_on_station_stream(src, st_comp, chan_win)
            results.append(measure)
            print("Number of measurements in station stream: %d"
                  % len(measure))
        except Exception as err:
            print("Failed to process data due to: %s" % err)

    return {"measure": results, "missing_stations": missing_stations}


def save_measurements(results, fn):
    data = {}
    for k in results[0]:
        data[k] = []

    for d in results:
        for k, v in d.iteritems():
            data[k].append(v)

    df = pd.DataFrame(data)
    print("Save features to file: %s" % fn)
    df.to_csv(fn)


def main():
    sources = pd.read_csv("../data/source.csv")
    sources.sort_values("time", ascending=False, inplace=True)

    # stations = pd.read_csv("../data/station.csv")

    waveform_base = "../../proc"
    window_base = "../../arrivals"

    nsources = len(sources)
    results = []
    missing_stations = 0
    for idx in range(nsources):
        src = sources.loc[idx]
        # if src.mag < 3.2:
        #    continue
        origin_time = obspy.UTCDateTime(src.time)
        print("=" * 10 + " [%d/%d]Source(%s, mag=%.2f, dep=%.2f km) "
              % (idx + 1, nsources, origin_time, src.mag, src.depth) +
              "=" * 10)

        waveform_file = os.path.join(
            waveform_base, "%s" % origin_time, "CI.mseed")
        window_file = os.path.join(window_base, "%s.json" % origin_time)

        _m = measure_on_stream(src, waveform_file, window_file)
        if _m is not None:
            results.extend(_m["measure"])
            missing_stations += _m["missing_stations"]
        print(" *** Missing stations in total: %d ***" % missing_stations)

    save_measurements(results, "../../measurements.csv")


if __name__ == "__main__":
    main()
