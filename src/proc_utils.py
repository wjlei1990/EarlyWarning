"""
Util functions for data processing
"""
import obspy
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel


def highpass_filter_waveform(trace, freq):
    trace.detrend('linear')
    trace.taper(0.05)
    trace.filter('highpass', freq=freq, corners=4, zerophase=False)
    # trace.detrend('linear')
    # trace.taper(0.05)


def lowpass_filter_waveform(trace, freq):
    trace.detrend('linear')
    trace.taper(0.05)
    trace.filter('lowpass', freq=freq, corners=4, zerophase=False)
    # trace.detrend('linear')
    # trace.taper(0.05)


def sort_station_by_epicenter_distance(src, stations):
    dists = []
    for idx in range(len(stations)):
        _d = locations2degrees(
            src["latitude"], src["longitude"],
            stations.loc[idx].latitude, stations.loc[idx].longitude)
        dists.append(_d)

    stations["distance"] = dists
    stations = stations.sort_values(by=["distance"])
    stations = stations.reset_index(drop=True)
    return stations


def filter_stream_by_sampling_rate(stream, threshold=19):
    stream_filter = obspy.Stream()
    for tr in stream:
        if tr.stats.sampling_rate < threshold:
            continue
        stream_filter.append(tr)

    print("Number of traces change: %d --> %d"
          % (len(stream), len(stream_filter)))
    return stream_filter


def get_predicted_first_arrival(src, dists):
    print("source depth: %.2f km" % src.depth)
    model = TauPyModel(model="prem")
    arrivals = []
    for deg in dists:
        arrivs = model.get_travel_times(
            src.depth, deg, phase_list=("p", "P", "Pn"))
        arrivals.append(arrivs[0].time)
    return arrivals
