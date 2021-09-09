import h5py
import numpy as np

with h5py.File("test.h5", "r") as ipt:
    DetectedElectrons = ipt["DetectedElectrons"][()]

DetectedElectrons = DetectedElectrons[DetectedElectrons["ImageId"] < 10]

Nele_this = len(DetectedElectrons)

A = np.random.normal(100, 10, Nele_this)
A = np.clip(A, a_min=0, a_max=None)
σ = np.random.normal(5e-3, 9e-4, Nele_this)
σ = np.clip(σ, a_min=0, a_max=None)
ρ = np.random.normal(0, 0, Nele_this)
ρ = np.clip(ρ, a_min=-1, a_max=1)
# print(A)
# print(σ)
print(ρ)


def ElectronSpot(x, y, A, σ, ρ) :
    return A * np.exp(-(x ** 2 + y ** 2) / (2 * σ ** 2))


Xgrid = np.linspace(-1, 1, 2048 + 1)
Zgrid = Xgrid
grid_area = (Xgrid[1] - Xgrid[0]) * (Zgrid[1] - Zgrid[0])
# when distance between screen and electron > rmax, light must be 0
rmaxs = np.sqrt(2 * np.log(A)) * σ
print(rmaxs)
x0s = DetectedElectrons["x"]
z0s = DetectedElectrons["z"]
# A rectengular zone for calculation
xmins = Xgrid[np.digitize(x0s - rmaxs, Xgrid, right=False) - 1]
xmaxs = Xgrid[np.digitize(x0s + rmaxs, Xgrid, right=True)]
zmins = Zgrid[np.digitize(z0s - rmaxs, Zgrid, right=False) - 1]
zmaxs = Zgrid[np.digitize(z0s + rmaxs, Zgrid, right=True)]
print(x0s)
print(xmaxs)

Xfinegrids = np.linspace(xmins, xmaxs, 100)  # axis 0: grids; axis 1: diferent electrons
Zfinegrids = np.linspace(zmins, zmaxs, 100)
finegrids = [np.meshgrid(Xfinegrids[:, i], Zfinegrids[:, i]) for i in range(Xfinegrids.shape[1])]
Xfinegrids = np.vstack([finegrids[i][0].flatten() for i in range(len(finegrids))]).T
Zfinegrids = np.vstack([finegrids[i][1].flatten() for i in range(len(finegrids))]).T

finegrid_areas = (Xfinegrids[1] - Xfinegrids[0]) * (Zfinegrids[1] - Zfinegrids[0])

Intensity = ElectronSpot(Xfinegrids - x0s, Zfinegrids - z0s, A, σ, ρ)
print(Intensity.sum(axis=0))
print(Intensity.max(axis=0))
from matplotlib import pyplot as plt
plt.hist2d(Xfinegrids.flatten(), Zfinegrids.flatten(), bins=1024, range=[[-1, 1], [-1, 1]], weights=Intensity.flatten())
plt.show()
