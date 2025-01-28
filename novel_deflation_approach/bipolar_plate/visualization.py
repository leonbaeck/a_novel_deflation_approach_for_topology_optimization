import cashocs
from dolfin import *
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

n = 75
mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(
    n, diagonal="crossed"
)

CG1 = FunctionSpace(mesh, "CG", 1)
phi = Function(CG1)

colors = np.array([[0, 107, 164], [255, 128, 14]]) / 255.0
cmap = LinearSegmentedColormap.from_list("tab10_colorblind", colors, N=256)

cvals = [74., 87., 100.]
color_line = [colors[0], "white", colors[1]]
norm = plt.Normalize(min(cvals), max(cvals))
tuples = list(zip(map(norm, cvals), color_line))
cmap2 = LinearSegmentedColormap.from_list("", tuples)

min_list = [
    0, 6, 8, 12, 15, 16, 18, 22, 23, 24, 25, 26, 29, 33, 34, 38, 41, 43, 46, 48, 50, 55,
    56, 61, 64, 70, 73, 74, 81, 83, 84, 85, 86, 90, 93, 94, 95, 100, 102, 108, 109, 114,
    117, 118, 119, 125, 129, 130, 132, 136, 141, 144, 148, 160, 165, 166, 170, 180, 181,
    183, 185, 189, 198, 200
]

it_deflation_list = []
it_total_list = []
vals_min = []

file_obj = open('./results/objective.txt', 'r')
lines_val = file_obj.readlines()
file_it_deflation = open('./results/iterations_deflation.txt', 'r')
lines_it_deflation = file_it_deflation.readlines()
file_it_total = open('./results/iterations_total.txt', 'r')
lines_it_total = file_it_total.readlines()
file_obj.close()
file_it_deflation.close()
file_it_total.close()

limit = 201
for i in range(0, limit):
    phi = Function(CG1)
    file_phi = HDF5File(mesh.mpi_comm(), './results/xdmf/phi_{d}.h5'.format(d=i), 'r')
    file_phi.read(phi, 'phi_{d}'.format(d=i))
    file_phi.close()

    constraint_val = float(lines_val[i])

    loc = (constraint_val - 74.) / 26.
    new_color = cmap2(loc)

    j = i
    if i in min_list:
        j = min_list.index(i) + 1
        vals_min.append(100. - constraint_val)

        plot(phi, vmin=-1e-7, vmax=1e-7, cmap=cmap)
        plt.xlabel(
            '{nr}: {val}%'.format(nr=j, val=constraint_val), fontsize=40,
            fontweight='bold', color='black',
            bbox={'fc': new_color, 'ec': 'white', 'alpha': 0.6}
        )
        plt.xticks([])
        plt.yticks([])
        plt.savefig(
            './results/phi/phi_{d}.png'.format(d=i), bbox_inches='tight',
            transparent=True
        )
        plt.close()

    it_deflation_list.append(int(lines_it_deflation[i]))
    it_total_list.append(int(lines_it_deflation[i]) + int(lines_it_total[i]))

plt.style.use("tableau-colorblind10")

y_array = range(0, limit)
plt.bar(y_array, it_total_list,  label='Total', width=1.0)
plt.bar(y_array, it_deflation_list,  label='Deflation', width=1.0)
plt.ylabel('Number of Iterations', fontweight='bold', fontsize=10)
plt.xlabel('Deflation iteration', fontweight='bold', fontsize=10)
plt.hlines(y=50., xmin=0, xmax=200., colors='black', linestyles='dotted', lw=1)
plt.hlines(y=100., xmin=0, xmax=200., colors='black', linestyles='dotted', lw=1)
plt.hlines(y=150., xmin=0, xmax=200., colors='black', linestyles='dotted', lw=1)
plt.hlines(y=200., xmin=0, xmax=200., colors='black', linestyles='dotted', lw=1)
plt.hlines(y=250., xmin=0, xmax=200., colors='black', linestyles='dotted', lw=1)
plt.hlines(y=300., xmin=0, xmax=200., colors='black', linestyles='dotted', lw=1)
plt.hlines(y=350., xmin=0, xmax=200., colors='black', linestyles='dotted', lw=1)
plt.hlines(y=400., xmin=0, xmax=200., colors='black', linestyles='dotted', lw=1)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.legend(loc="upper right")
plt.savefig('./results/iterations.png', bbox_inches='tight', transparent=True)
plt.close()

y_array = range(0, len(min_list))
plt.plot(y_array, vals_min, color=colors[1], marker='.')
plt.yscale('log')
plt.ylabel('Constraint gap [%]', fontweight='bold', fontsize=10)
plt.xlabel('Counter of local minimizer', fontweight='bold', fontsize=10)
plt.hlines(y=10., xmin=0, xmax=63, colors='black', linestyles='dotted', lw=1)
plt.hlines(y=1., xmin=0, xmax=63, colors='black', linestyles='dotted', lw=1)
plt.hlines(y=0.1, xmin=0, xmax=63, colors='black', linestyles='dotted', lw=1)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.ylim(0.025, 100)
plt.savefig("./results/objective_percentage.png", bbox_inches='tight', transparent=True)
plt.close()
