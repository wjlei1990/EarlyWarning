"""
1) use the arrival time to generate windows
2) use the window and seismogram to generate measurements(inside windows)
"""
from __future__ import print_function, division
import os
import sys  # NOQA
import numpy as np
import h5py
import pandas as pd
import obspy
import json
from obspy import UTCDateTime
import matplotlib.pyplot as plt


def load_json(fn):
    with open(fn) as fh:
        return json.load(fh)


def read_waveform(fn):
    st = obspy.read(fn, format="MSEED")
    return st


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


def extract_station_stream(st, chan_win, interp_npts, interp_sampling_rate):
    arrival = UTCDateTime(chan_win["pick_arrival"])
    data = {}

    for tr in st:
        tr.interpolate(interp_sampling_rate, starttime=arrival-0.25,
                       npts=interp_npts)
        if tr.stats.npts != interp_npts:
            raise ValueError("Error interpolating trace: %s" % tr.id)
        data[tr.stats.channel[-1]] = tr.data

    return data


def extract_on_stream(src, waveform_file, window_file,
                      interp_npts, interp_sampling_rate):
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

    dataset = []
    mags = []
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
            print(st_comp)
            station_data = extract_station_stream(
                st_comp, chan_win, interp_npts, interp_sampling_rate)
            if len(station_data) == 3:
                dataset.append(station_data)
        except Exception as err:
            print("Failed to process data due to: %s" % err)

    mags = [src.mag, ] * len(dataset)
    print("Number of stations include: %d" % len(dataset))
    return {"data": dataset, "missing_stations": missing_stations,
            "magnitude": mags}


def save_data(data, magnitude, interp_npts, outputfn):
    print("Number of measurements: %d" % len(data))
    print("Number of magnitudes: %d" % len(magnitude))
    if len(data) != len(magnitude):
        raise ValueError("Length mismatch between data and magnitude:"
                         "%d, %d" % (len(d), len(magnitude)))

    array = np.zeros([len(data), 3, interp_npts])
    for idx, d in enumerate(data):
        array[idx, 0, :] = d["Z"]
        array[idx, 1, :] = d["N"]
        array[idx, 2, :] = d["E"]

    f = h5py.File(outputfn, 'w')
    dset = f.create_dataset("waveform", data=array)
    mset = f.create_dataset("magnitude", data=magnitude)
    f.close()


def main():
    sources = pd.read_csv("../data/source.csv")
    sources = sources.sort_values("time", ascending=True)
    sources = sources.reset_index()

    # stations = pd.read_csv("../data/station.csv")

    waveform_base = "../../proc"
    window_base = "../../arrivals"

    interp_npts = 401
    interp_sampling_rate = 20

    nsources = len(sources)
    data = []
    magnitude = []
    for idx in range(nsources):
    #for idx in range(1000):
        src = sources.loc[idx]
        # if src.mag < 3.2:
        #    continue
        origin_time = UTCDateTime(src.time)
        print("=" * 10 + " [%d/%d]Source(%s, mag=%.2f, dep=%.2f km) "
              % (idx + 1, nsources, origin_time, src.mag, src.depth) +
              "=" * 10)

        waveform_file = os.path.join(
            waveform_base, "%s" % origin_time, "CI.mseed")
        window_file = os.path.join(window_base, "%s.json" % origin_time)

        _d = extract_on_stream(src, waveform_file, window_file,
                               interp_npts, interp_sampling_rate)
        if _d is not None and len(_d["data"]) > 0:
            data.extend(_d["data"])
            magnitude.extend(_d["magnitude"])

    save_data(data, magnitude, interp_npts, "test.h5")


if __name__ == "__main__":
    main()
