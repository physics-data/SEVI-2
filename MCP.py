import h5py
import numpy as np

with h5py.File("test.h5", "r") as ipt:
    DetectedElectrons = ipt["DetectedElectrons"][()]

DetectedElectrons = DetectedElectrons[DetectedElectrons["ImageId"] == 0]

Nele_this = len(DetectedElectrons)

A = np.random.normal(100, 10, Nele_this)
A = np.clip(A, min=0)
σ = np.random.normal(1e-3, 1e-4, Nele_this)
σ = np.clip(σ, min=0)
ρ = np.random.normal(0, 1, Nele_this)


def ElectronSpot(x, y, A, σ, ρ) :
    return A * np.exp((x ** 2 + y ** 2 + 2 * ρ * x * y) / (2 * σ ** 2))


Xgrid = np.linspace(-1, 1, 2048 + 1)
Zgrid = Xgrid
grid_area = (Xgrid[1] - Xgrid[0]) * (Zgrid[1] - Zgrid[0])
## when distance between screen and electron > rmax, light must be 0
rmaxs = np.sqrt(2 * np.log(A)) * σ
x0s = DetectedElectrons["x"]
z0s = DetectedElectrons["z"]
## A rectengular zone for calculation
xmins = Xgrid[np.digitize(x0s - rmaxs, Xgrid, right=False) - 1]
xmaxs = Xgrid[np.digitize(x0s + rmaxs, Xgrid, right=True)]
zmins = Zgrid[np.digitize(z0s - rmaxs, Zgrid, right=False) - 1]
zmaxs = Zgrid[np.digitize(z0s + rmaxs, Zgrid, right=True)]

Xfinegrids = np.linspace(xmins, xmaxs, 100)  # axis 0: grids; axis 1: diferent electrons
Zfinegrids = np.linspace(zmins, zmaxs, 100)
finegrid_areas = (Xfinegrids[1] - Xfinegrids[0]) * (Zfinegrids[1] - Zfinegrids[0])

Intensity = ElectronSpot(Xfinegrids, Zfinegrids, A, σ, ρ)
