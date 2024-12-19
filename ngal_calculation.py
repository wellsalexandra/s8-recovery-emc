# compute number density of galaxies

# Simulation A:
galaxies = np.load('galaxies_a.npy')
x = galaxies[:, 0]
y = galaxies[:, 1]
z = galaxies[:, 2]

xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()
zmin, zmax = z.min(), z.max()

Lx = xmax - xmin
Ly = ymax - ymin
Lz = zmax - zmin
volume = Lx * Ly * Lz

ngal_a = len(x)/volume

galaxies = np.load('galaxies_b.npy')
x = galaxies[:, 0]
y = galaxies[:, 1]
z = galaxies[:, 2]

xmin, xmax = x.min(), x.max()
ymin, ymax = y.min(), y.max()
zmin, zmax = z.min(), z.max()

Lx = xmax - xmin
Ly = ymax - ymin
Lz = zmax - zmin
volume = Lx * Ly * Lz

ngal_b = len(x)/volume

ngal = np.mean([ngal_a, ngal_b], axis = 0)

# with error of 1% of the measurement:
sigma_ngal=0.01*ngal
# giving a 1x1 covariance matrix of sigma_ngla**2

table = Table()
table['ngal'] = [ngal]
table['ngal_err'] = [sigma_ngal]
table.write('ngal.csv', overwrite=True)


