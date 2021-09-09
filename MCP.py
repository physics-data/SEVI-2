import h5py
import numpy as np
from time import time

with h5py.File("test.h5", "r") as ipt:
    DetectedElectrons = ipt["DetectedElectrons"][()]

DetectedElectrons = DetectedElectrons[DetectedElectrons["ImageId"] == 0]

Nele_this = len(DetectedElectrons)

A = np.random.normal(100, 10, Nele_this)
A = np.clip(A, a_min=0, a_max=None)
σ = np.random.normal(1.5e-2, 1e-3, Nele_this)
σ = np.clip(σ, a_min=0, a_max=None)
ρ = np.random.normal(0, 0.1, Nele_this)
ρ = np.clip(ρ, a_min=-1, a_max=1)
# print(A)
# print(σ)
print(ρ)


def ElectronSpot(x, y, A, σ, ρ) :
    return A * np.exp(-(x ** 2 + y ** 2 + 2 * ρ * x * y) / (2 * σ ** 2))


size = 1024
Xgrid = np.linspace(-1, 1, size + 1)
Zgrid = Xgrid
Finess = 5
FineGrids = np.linspace(-1, 1, size * Finess + 1)
start = time()
# when distance between screen and electron > rmax, light must be 0
rmaxs = np.sqrt(2 * np.log(A)) * σ
print(rmaxs)
x0s = DetectedElectrons["x"]
z0s = DetectedElectrons["z"]
# A rectengular zone for calculation
xmins = (np.digitize(x0s - rmaxs, Xgrid, right=False) - 1) * Finess
xmaxs = (np.digitize(x0s + rmaxs, Xgrid, right=True)) * Finess
zmins = (np.digitize(z0s - rmaxs, Zgrid, right=False) - 1) * Finess
zmaxs = (np.digitize(z0s + rmaxs, Zgrid, right=True)) * Finess

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
theImage += np.random.normal(0, 10, (size, size))
print(time() - start)
print(A.max())
print(theImage.max())
theIm = np.round(theImage).astype(np.uint8)
from matplotlib import pyplot as plt
plt.imshow(theImage)
# plt.hist2d(Xfinegrids, Zfinegrids, bins=size, range=[[-1, 1], [-1, 1]], weights=Intensity)
plt.show()
