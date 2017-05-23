import os
import h5py
import numpy as np


f = h5py.File("./test.h5")
waves = np.array(f["waveform"])
mags = np.array(f["magnitude"])
dists = np.array(f["distance"])
f.close()

print("input waves, dists, and mags shape: ", waves.shape,
      dists.shape, mags.shape)

# deal with large earthquakes first
msk = mags < 6.0
w1 = waves[msk]
m1 = mags[msk]
d1 = dists[msk]
w2 = waves[~msk]
m2 = mags[~msk]
d2 = dists[~msk]
print("number of large mags: %d" % (len(m2)))

# split the data
msk = np.random.rand(len(m1)) < 0.8
train_x = np.concatenate((w1[msk], w2[0:1, :]), axis=0)
train_y = np.concatenate((m1[msk], m2[0:1]), axis=0)
train_d = np.concatenate((d1[msk], d2[0:1]), axis=0)
test_x =  np.concatenate((w1[~msk], w2[1:, :]), axis=0)
test_y =  np.concatenate((m1[~msk], m2[1:]), axis=0)
test_d =  np.concatenate((d1[~msk], d2[1:]), axis=0)

print("train x and y shape: ", train_x.shape, train_y.shape)
print("test x and y shape: ", test_x.shape, test_y.shape)
print("train d and test d shape: ", train_d.shape, test_d.shape)

outputfn = "data.h5"
if os.path.exists(outputfn):
    os.remove(outputfn)

n1 = (train_y >= 5).sum()
n2 = (train_y >= 6).sum()
print("Number of events(mag>5, mag>6) in train: %d, %d" % (n1, n2))

n1 = (test_y >= 5).sum()
n2 = (test_y >= 6).sum()
print("Number of events(mag>5, mag>6) in test: %d, %d" % (n1, n2))

f = h5py.File(outputfn, 'w')
f.create_dataset("train_x", data=train_x)
f.create_dataset("train_y", data=train_y)
f.create_dataset("test_x", data=test_x)
f.create_dataset("test_y", data=test_y)
f.create_dataset("train_distance", data=train_d)
f.create_dataset("test_distance", data=test_d)
f.close()
