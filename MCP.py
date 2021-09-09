import argparse
import sys
psr = argparse.ArgumentParser()
psr.add_argument('ipt', help='input')
psr.add_argument('opt', help='output')
psr.add_argument('-t', '--with-truth', dest='withtruth', action='store_true')
psr.add_argument('-b', '--break-at', dest='stop', type=int, default=-1)
args = psr.parse_args()

import h5py
import numpy as np
import pandas as pd

with h5py.File(args.ipt, "r") as ipt:
    DetectedElectrons = ipt["DetectedElectrons"][()]

# DetectedElectrons = DetectedElectrons[DetectedElectrons["ImageId"] == 0]

Nele_this = len(DetectedElectrons)

A = np.random.normal(100, 10, Nele_this)
A = np.clip(A, a_min=0, a_max=None)
σ = np.random.normal(1.5e-2, 1e-3, Nele_this)
σ = np.clip(σ, a_min=0, a_max=None)
ρ = np.random.normal(0, 0.1, Nele_this)
ρ = np.clip(ρ, a_min=-1, a_max=1)

eleDF = pd.DataFrame.from_records(DetectedElectrons)
eleDF["A"] = A
eleDF["σ"] = σ
eleDF["ρ"] = ρ


size = 1024
Xgrid = np.linspace(-1, 1, size + 1)
Zgrid = Xgrid
Finess = 5
FineGrids = np.linspace(-1, 1, size * Finess + 1)


def ElectronSpot(x, y, A, σ, ρ) :
    return A * np.exp(-(x ** 2 + y ** 2 + 2 * ρ * x * y) / (2 * σ ** 2))


def GenerateOneImage(x0s, z0s, A, σ, ρ) :
    # when distance between screen and electron > rmax, light must be 0
    rmaxs = np.sqrt(2 * np.log(A)) * σ
    # A rectengular zone for calculation
    xmins = (np.digitize(x0s - rmaxs, Xgrid, right=False) - 1) * Finess
    xmaxs = (np.digitize(x0s + rmaxs, Xgrid, right=True)) * Finess
    xmaxs = np.clip(xmaxs, a_min=None, a_max=size * Finess)
    zmins = (np.digitize(z0s - rmaxs, Zgrid, right=False) - 1) * Finess
    zmaxs = (np.digitize(z0s + rmaxs, Zgrid, right=True)) * Finess
    zmaxs = np.clip(zmaxs, a_min=None, a_max=size * Finess)

    Intensity = []
    Xfinegrids = []
    Zfinegrids = []
    for i in range(len(x0s)) :
        Xfinegrid, Zfinegrid = np.meshgrid(np.arange(xmins[i], xmaxs[i] + 1), np.arange(zmins[i], zmaxs[i] + 1))
        Xfinegrid = FineGrids[Xfinegrid.flatten()]
        Zfinegrid = FineGrids[Zfinegrid.flatten()]
        Xfinegrids.append(Xfinegrid)
        Zfinegrids.append(Zfinegrid)
        Intensity.append(ElectronSpot(Xfinegrid - x0s[i], Zfinegrid - z0s[i], A[i], σ[i], ρ[i]))

    Intensity = np.concatenate(Intensity)
    Xfinegrids = np.concatenate(Xfinegrids)
    Zfinegrids = np.concatenate(Zfinegrids)
    theImage = np.histogram2d(Xfinegrids, Zfinegrids, bins=size, range=[[-1, 1], [-1, 1]], weights=Intensity)[0] / Finess ** 2
    theImage += np.random.normal(20, 20, (size, size))
    theImage = np.clip(np.round(theImage), a_min=0, a_max=255).astype(np.uint8)
    return theImage


Nimages_in = DetectedElectrons["ImageId"].max() + 1
Nimage_out = Nimages_in if args.stop == -1 else min(args.stop, Nimages_in)
Image_dt = np.dtype([("ImageId", np.int32), ("Image", np.uint8, (size, size))])
flush_num = 100
ImageGroup = eleDF.groupby("ImageId")
with h5py.File(args.opt, "w") as opt :
    opt.create_dataset("FinalImage", shape=(Nimage_out,), dtype=Image_dt)
    if args.withtruth :
        with h5py.File(args.ipt, "r") as ipt :
            for keyname in list(ipt.keys()) :
                opt.create_dataset(keyname, data=ipt[keyname])

    counter = 0
    batch_counter = 0
    FinalImages = np.empty(flush_num, dtype=Image_dt)
    for ImageId, ThisImageElec in ImageGroup :
        if batch_counter * flush_num + counter == args.stop : break
        if counter == flush_num :
            opt["FinalImage"][batch_counter * flush_num:batch_counter * flush_num + counter] = FinalImages  # flush data to disk
            counter = 0
            batch_counter += 1
        FinalImages["Image"][counter] = GenerateOneImage(ThisImageElec["x"].values, ThisImageElec["z"].values, ThisImageElec["A"].values, ThisImageElec["σ"].values, ThisImageElec["ρ"].values)
        print("ImageId={0}, counter={1}, batch_counter={2}".format(ImageId, counter, batch_counter))
        FinalImages["ImageId"][counter] = ImageId
        counter += 1
    opt["FinalImage"][batch_counter * flush_num:batch_counter * flush_num + counter] = FinalImages[:counter]  # flush data to disk
