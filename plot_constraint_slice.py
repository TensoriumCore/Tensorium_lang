#!/usr/bin/env python3

import csv
import sys

import matplotlib.pyplot as plt
import numpy as np


if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} solution.csv [field]")
    sys.exit(1)

csv_path = sys.argv[1]
field = sys.argv[2] if len(sys.argv) > 2 else "conformal_factor"

with open(csv_path, newline="") as stream:
    rows = list(csv.DictReader(stream))

if not rows or field not in rows[0]:
    print(f"Unknown field: {field}")
    sys.exit(1)

# Plot the first spectral domain so compactified infinity does not flatten
# the interesting part of the solution.
domain = rows[0]["domain"]
samples = [
    (float(row["r"]), float(row[field]))
    for row in rows
    if row["domain"] == domain and np.isfinite(float(row["r"]))
]
samples.sort()
radii = np.array([sample[0] for sample in samples])
values = np.array([sample[1] for sample in samples])

axis = np.linspace(-radii[-1], radii[-1], 250)
x, y = np.meshgrid(axis, axis)
r = np.sqrt(x**2 + y**2)
z = np.interp(r, radii, values)
z[(r < radii[0]) | (r > radii[-1])] = np.nan

figure = plt.figure()
plot = figure.add_subplot(111, projection="3d")
plot.plot_surface(x, y, z, cmap="viridis")
plot.set_xlabel("x")
plot.set_ylabel("y")
plot.set_zlabel(field)
plot.set_title(f"{field} - {domain}")
plt.show()
