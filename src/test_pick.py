import os
import obspy
from obspy import UTCDateTime
from obspy.taup import TauPyModel
import pandas as pd
from arrival_pick import p_wave_onset_and_SNR, plot_stream_and_arrival
from check_data_quality import sort_by_epicenter_distance


def get_predicted_first_arrival(src, dists):
    model = TauPyModel(model="prem")
    arrivals = []
    for deg in dists:
        arrivs = model.get_travel_times(
            8.0, deg, phase_list=("p", "P", "Pn"))
        arrivals.append(arrivs[0].time)
    return arrivals


def pick_one_stream(src, stations, st):
    st = st.select(channel="BHZ")

    stations = sort_by_epicenter_distance(src, stations)
    stations = stations.reset_index(drop=True)

    picks = {}
    dists = []
    idx_picks = 0
    st_new = obspy.Stream()
    for i in range(len(stations)):
        _nw = stations.loc[i].network
        _sta = stations.loc[i].station
        _dist = stations.loc[i].distance
        _st = st.select(network=_nw, station=_sta)
        if len(_st) == 0:
            #print("Missing station: %s.%s -- %.5f"
            #      % (_nw, _sta, _dist))
            continue
        for tr in _st:
            idx_picks += 1
            if idx_picks > 12:
                break
            print("%d - [%s] -- %.5f" % (idx_picks, tr.id, _dist))
            _pick = p_wave_onset_and_SNR(
                tr, UTCDateTime(src.time), SNR_plot_flag=False,
                trigger_plot_flag=False)
            picks[tr.id] = _pick
            st_new.append(tr)
            dists.append(_dist)

    arrivals = get_predicted_first_arrival(src, dists)
    print("arrivals: %s" % arrivals)
    plot_stream_and_arrival(st_new, picks, UTCDateTime(src.time),
                            arrivals)



def main():

    sources = pd.read_csv("../data/source.csv")
    sources.sort_values("time", ascending=False, inplace=True)

    stations = pd.read_csv("../data/station.csv")

    database = "../../proc"
    for idx in range(len(sources)):
        src = sources.loc[idx]
        origin_time = obspy.UTCDateTime(src.time)
        print("============= Source(%s, %.2f) ============="
              % (origin_time, src.mag))
        datafile = os.path.join(database, "%s" % origin_time, "CI.mseed")
        if not os.path.exists(datafile):
            contiue
        st = obspy.read(datafile)
        pick_one_stream(src, stations, st)


if __name__ == "__main__":
    main()
