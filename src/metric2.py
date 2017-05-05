import numpy as np
import obspy.signal


def norm(data, ord):
    return np.linalg.norm(data, ord=ord)


def envelope(data):
    return obspy.signal.filter.envelope(data)


def split_data(data, n=3):
    npts = len(data)
    npts_split = int(npts / n)

    data_split = []
    for i in range(n):
        _d = data[i*npts_split: (i+1)*npts_split]
        data_split.append(_d)

    return data_split
