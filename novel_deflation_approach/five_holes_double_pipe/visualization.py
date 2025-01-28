import cashocs
from dolfin import *
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(
    "./mesh/mesh_deflation.xdmf"
)

CG1 = FunctionSpace(mesh, "CG", 1)
phi = Function(CG1)

colors = np.array([[0, 107, 164], [255, 128, 14]]) / 255.0
cmap = LinearSegmentedColormap.from_list("tab10_colorblind", colors, N=256)

min_list = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 20, 22, 27, 29, 31, 34, 35,
    36, 42, 44, 46, 48, 50, 57, 67, 68, 74, 75, 79, 80, 86
]

limit = 101
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

for i in range(0, limit):
    phi = Function(CG1)
    file_phi = HDF5File(mesh.mpi_comm(), './results/xdmf/phi_{d}.h5'.format(d=i), 'r')
    file_phi.read(phi, 'phi_{d}'.format(d=i))
    file_phi.close()

    obj_val = float(lines_val[i])

    j = i
    if i in min_list:
        j = min_list.index(i) + 1
        vals_min.append(obj_val)

        plot(phi, vmin=-1e-7, vmax=1e-7, cmap=cmap)
        plt.xlabel(
            '{i}: J={val}'.format(i=j, val=obj_val), fontsize=40,
            fontweight='bold'
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
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.hlines(y=50., xmin=0, xmax=100., colors='black', linestyles='dotted', lw=1)
plt.hlines(y=100., xmin=0, xmax=100., colors='black', linestyles='dotted', lw=1)
plt.hlines(y=150., xmin=0, xmax=100., colors='black', linestyles='dotted', lw=1)
plt.hlines(y=200., xmin=0, xmax=100., colors='black', linestyles='dotted', lw=1)
plt.hlines(y=250., xmin=0, xmax=100., colors='black', linestyles='dotted', lw=1)
plt.hlines(y=300., xmin=0, xmax=100., colors='black', linestyles='dotted', lw=1)
plt.legend(loc="upper right")
plt.savefig('./results/iterations.png', bbox_inches='tight', transparent=True)
plt.close()

y_array = range(0, len(min_list))
plt.plot(y_array, vals_min, marker='.', color=colors[1])
plt.ylabel('Objective value', fontweight='bold', fontsize=10)
plt.xlabel('Counter of local minimizer', fontweight='bold', fontsize=10)
plt.xticks(fontweight='bold')
plt.yticks(fontweight='bold')
plt.hlines(y=80., xmin=0, xmax=37., colors='black', linestyles='dotted', lw=1)
plt.hlines(y=90., xmin=0, xmax=37., colors='black', linestyles='dotted', lw=1)
plt.hlines(y=100., xmin=0, xmax=37., colors='black', linestyles='dotted', lw=1)
plt.hlines(y=110., xmin=0, xmax=37., colors='black', linestyles='dotted', lw=1)
plt.hlines(y=120., xmin=0, xmax=37., colors='black', linestyles='dotted', lw=1)
plt.savefig('./results/obj_vals.png', bbox_inches='tight', transparent=True)
plt.close()
