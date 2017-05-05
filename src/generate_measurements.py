"""
1) use the arrival time to generate windows
2) use the window and seismogram to generate measurements(inside windows)
"""
import os
import scipy.stats
import numpy as np
import pandas as pd
import obspy
from obspy import UTCDateTime
import matplotlib.pyplot as plt
from utils import load_json
from metric import cut_window, tau_c, tau_p_max, \
    make_disp_vel_acc_records
from metric2 import norm, envelope, split_data


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
    measure["%s.max_amp" % prefix] = np.max(np.abs(data))

    _, _, mean, var, skew, kurt = scipy.stats.describe(data)
    measure["%s.mean" % prefix] = mean
    measure["%s.var" % prefix] = var
    measure["%s.skew" % prefix] = skew
    measure["%s.kurt" % prefix] = kurt
    return measure


def measure_on_trace_data_type(trace, data_type):
    measure = {}

    prefix = data_type
    measure.update(measure_func(trace.data, prefix))

    data_split = split_data(trace.data, n=3)
    for idx in range(len(data_split)):
        prefix = "%s.window_%d" % (data_type, idx)
        measure.update(measure_func(data_split[idx], prefix))

    env_data = envelope(trace.data)
    prefix = "%s.env" % data_type
    measure.update(measure_func(env_data, prefix))

    env_split = split_data(env_data, n=3)
    for idx in range(len(data_split)):
        prefix = "%s.env.window_%d" % (data_type, idx)
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


def measure_on_trace(_trace, windows, src, win_len=3.0):
    arrival = UTCDateTime(windows["pick_arrival"])
    # plot_arrival_window(_trace, windows, UTCDateTime(src.time))

    trace = cut_window(_trace, arrival, win_len)
    print(trace)

    measure = {}
    _v = tau_p_max(trace)
    measure["tau_p_max"] = _v["tau_p_max"]

    trace_types = make_disp_vel_acc_records(trace)
    measure["tau_c"] = tau_c(trace_types["disp"], trace_types["vel"])

    for dtype, data in trace_types.iteritems():
        measure.update(measure_on_trace_data_type(data, dtype))

    measure["distance"] = windows["distance"]
    measure["channel"] = trace.id
    measure["source"] = "%s" % UTCDateTime(src.time)
    measure["magnitude"] = src.mag
    # print(measure)
    # print("Number of features: %d" % len(measure))
    # print(measure.keys())
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
    for chan, chan_win in windows.iteritems():
        try:
            trace = st.select(id=chan)[0]
            measure = measure_on_trace(trace, chan_win, src)
            results.append(measure)
        except Exception as err:
            print("Failed to process data due to: %s" % err)

    return results


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
    for idx in range(nsources):
        src = sources.loc[idx]

        # if src.depth < 0:
        #    print("Error source depth(%.2f km) < 0" % src.depth)
        #    nbad_source += 1
        #    continue
        # if src.mag < 3.2:
        #    continue

        origin_time = obspy.UTCDateTime(src.time)
        print("=" * 10 + " [%d/%d]Source(%s, mag=%.2f, dep=%.2f km) "
              % (idx + 1, nsources, origin_time, src.mag, src.depth) +
              "=" * 10)

        waveform_file = os.path.join(
            waveform_base, "%s" % origin_time, "CI.mseed")
        window_file = os.path.join(window_base, "%s.json" % origin_time)
        measurements = measure_on_stream(src, waveform_file, window_file)
        if measurements is not None:
            results.extend(measurements)

    save_measurements(results, "../../measurements.csv")


if __name__ == "__main__":
    main()
