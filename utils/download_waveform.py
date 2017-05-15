import os
import time
import pandas as pd
from obspy import UTCDateTime
from pytomo3d.utils.download import download_waveform


def check_no_dup(times):
    tset = set()
    for t in times:
        if t in tset:
            print("Duplicate time: %s" % t)
        tset.add(t)


def stats_stream(streams):
    for sta_id, st in streams.iteritems():
        if st is None:
            print("[%s] is None" % sta_id)
            continue

        stations = set()
        for tr in st:
            tag = "%s.%s" % (tr.stats.network, tr.stats.station)
            stations.add(tag)
        print("[%s]Number of stations and traces: %d, %d"
              % (sta_id, len(stations), len(st)))


def download_func(eventtime, outputbase, before=-60, after=120):
    t1 = time.time()

    outputdir = os.path.join(outputbase, "%s" % eventtime)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    station_list = ["CI_*", "AZ_*"]
    print("Download time: %f, %f" % (before, after))
    print("Station list: %s" % station_list)
    _data = download_waveform(
        station_list, eventtime + before, eventtime + after,
        outputdir=outputdir)
    stats_stream(_data["stream"])

    t2 = time.time()
    print("outputdir: %s -- time: %.2f sec" % (outputdir, t2 - t1))


def check_download_finish(datadir):
    fn = os.path.join(datadir, "CI_*.mseed")
    if os.path.exists(fn):
        return True
    else:
        return False


def main():
    outputbase = "output"
    if not os.path.exists(outputbase):
        os.makedirs(outputbase)

    sources = pd.read_csv("source/query.csv")
    check_no_dup(sources.time)

    nevents = len(sources)
    print("Number of sources: %d" % nevents)
    for idx, ev in sources.iterrows():
        if idx != (nevents - 1):
            nextdir = os.path.join(
                outputbase, "%s" % UTCDateTime(sources.loc[idx+1].time))
            if check_download_finish(nextdir):
                continue

        t = UTCDateTime(ev.time)
        print(" ===== [idx: %d/%d] -- time: %s ===== " % (idx, nevents, t))
        download_func(t, outputbase)


if __name__ == "__main__":
    main()
