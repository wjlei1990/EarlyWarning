import os
import pandas as pd
import obspy
from obspy.geodetics import locations2degrees


def sort_by_epicenter_distance(src, stations):
    dists = []
    for idx in range(len(stations)):
        _d = locations2degrees(
            src["latitude"], src["longitude"],
            stations.loc[idx].latitude, stations.loc[idx].longitude)
        dists.append(_d)

    stations["distance"] = dists
    stations = stations.sort_values(by=["distance"])
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



def check_data_quality(src, stations, datadir):
    stations = stations.copy()
    stations = filter_by_epicenter_distance(src, stations)

    waveform_file = os.path.join(datadir, "CI_*.mseed")
    stream = obspy.read(waveform_file)
    stream = filter_stream_by_sampling_rate(stream)

    nstations_pick = 10
    ncount = 0
    for i in range(len(stations)):
        nw = stations.loc[i].network
        sta = stations.loc[i].station
        _st = stream.select(network=nw, station=sta)
        if len(_st) > 0:
            ncount += 1
            _st.plot()
            
        if ncount > 3:
            break


def main():
    stations = pd.read_csv("staxml/STATIONS.csv")
    print("Number of stations: %d" % len(stations))

    content = load_txt("./source/hs_1981_2016_comb_K4_A.cat_so_scsn_v2q") 
    content = content[::-1]
    n_finish = 0
    for line in content:
        src = Source.from_string(line)
        datadir = os.path.join("waveform", "%s" % src.starttime)
        if os.path.exists(datadir):
            print("=" * 20)
            print("Event dir: %s" % datadir)
            print("Magnitude: %.2f" % src.magnitude)
            n_finish += 1
            check_data_quality(src, stations, datadir)

    print("Number of events finished: %d" % n_finish)



if __name__ == "__main__":
    main()
