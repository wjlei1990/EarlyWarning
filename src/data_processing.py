from __future__ import print_function, division
import os
import time
import glob
import obspy

# keep channels sampling_rate > 20Hz
# detrend, demean, taper
# filter, highpass, 0.075Hz
# divide by the gain


def select_by_sampling_rate(st, threshold=15):
    st_new = obspy.Stream()
    for tr in st:
        if tr.stats.sampling_rate < threshold:
            continue
        st_new.append(tr)

    print("Number of traces change(>%d Hz): %d --> %d"
          % (threshold, len(st), len(st_new)))
    return st_new


def filter_waveform(st, freq=0.075, taper_percentage=0.05, taper_type="hann"):
    st.detrend("linear")
    st.detrend("demean")
    st.taper(max_percentage=taper_percentage, type=taper_type)

    for tr in st:
        tr.filter("highpass", freq=freq)

    return st


def remove_instrument_gain(st, invs):
    ntr1 = len(st)
    n_missing_inv = 0
    print("Number of inventories: %d" % len(invs))

    st_new = obspy.Stream()
    for tr in st:
        nw = tr.stats.network
        sta = tr.stats.station
        tag = "%s_%s" % (nw, sta)
        if tag not in invs:
            # print("Missing station: %s" % tag)
            n_missing_inv += 1
            continue
        inv = invs[tag].select(
            network=nw, station=sta, location=tr.stats.location,
            channel=tr.stats.channel,
            starttime=tr.stats.starttime, endtime=tr.stats.endtime)
        if len(inv) == 0:
            # print("Missing inventory: %s" % tr.id)
            continue
        try:
            sens = inv[0][0][0].response.instrument_sensitivity.value
            tr.data /= sens
            st_new.append(tr)
        except Exception as err:
            print("Error remove instrument gain(%s): %s" % (tr.id, err))

    ntr2 = len(st_new)
    print("Number of traces missing inventory: %d" % n_missing_inv)
    print("Number of trace change after remove instrument gain: %d --> %d"
          % (ntr1, ntr2))
    return st_new


def process_one_file(fn, invs, outputfn):
    print("-" * 10 + "Process file: %s" % fn, "-" * 10)
    st = obspy.read(fn)
    st = select_by_sampling_rate(st)
    st = filter_waveform(st)
    # remove instrument gain
    st = remove_instrument_gain(st, invs)

    print("Saved streams to file: %s" % outputfn)
    st.write(outputfn, format="MSEED")


def process_one_event(dirname, invs, outputdir):
    print("outputdir: %s" % outputdir)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    fn = os.path.join(dirname, "CI_*.mseed")
    if os.path.exists(fn):
        outputfn = os.path.join(outputdir, "CI.mseed")
        try:
            process_one_file(fn, invs, outputfn)
        except Exception as err:
            print("Error processing file(%s) due to: %s" % (fn, err))

    fn = os.path.join(dirname, "AZ_*.mseed")
    if os.path.exists(fn):
        outputfn = os.path.join(outputdir, "AZ.mseed")
        try:
            process_one_file(fn, invs, outputfn)
        except Exception as err:
            print("Error processing file(%s) due to: %s" % (fn, err))


def read_all_inventories():
    files = glob.glob("station_xml/*.xml")
    nfiles = len(files)

    invs = {}
    t1 = time.time()
    for idx, f in enumerate(files):
        _t1 = time.time()
        inv = obspy.read_inventory(f)
        _t2 = time.time()
        tag = os.path.basename(f).split(".")[0]
        invs[tag] = inv
        print("[%d/%d]Read staxml(%s): %.2f sec" %
              (idx+1, nfiles, tag, _t2 - _t1))
    t2 = time.time()

    print("Time reading inventories: %.2f sec" % (t2 - t1))
    return invs


def main():
    invs = read_all_inventories()
    # invs = {"CI_ADO": obspy.read_inventory("./station_xml/CI_ADO.xml")}
    outputbase = "proc"

    dirnames = glob.glob("output/*")
    ndirs = len(dirnames)
    print("Number of dirs: %d" % ndirs)
    dirnames = sorted(dirnames, reverse=True)
    for idx, _dir in enumerate(dirnames):
        print("=" * 15 + " [%d/%d] dir: %s " % (idx+1, ndirs, _dir) + "=" * 15)
        outputdir = os.path.join(outputbase, os.path.basename(_dir))
        process_one_event(_dir, invs, outputdir)


if __name__ == "__main__":
    main()
