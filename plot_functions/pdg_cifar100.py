import matplotlib.pyplot as plt
import numpy as np

x_rob = np.array([4,5,6, 10,11,12, ])
x_car = np.array([16,17,18, 24,25,26])
depth_rob = np.array([13,13,7, 17,17,8, ])
depth_car = np.array([13,12,8,16,17,8])

width_rob = np.array([5,7,3, 8,9,4])
width_car = np.array([8,6,3, 8,7,4])

fig, ax = plt.subplots()
ax1 = ax.twinx()

ax.bar(x_rob, depth_rob, color='blue', label='rob')
ax.bar(x_rob, -width_rob, color='orange', label='rob')
ax.bar(x_car, depth_car, color='red', label='car')
ax.bar(x_car, -width_car, color='orange', label='car')
ax.yaxis.set_major_formatter(lambda x, pos: f'{abs(x):g}')
ax.margins(x=0.1)
ax.set_xticks([5,11,17,25])
ax.set_xticklabels(['5G','10G','5G','10G'])
ax.set_xlabel('FLOPs')
ax.set_yticks([-20,-10,0,10,20])
ax.set_ylabel(['W_i','D_i'])

x = np.array([5,11,17,25])
y = np.array([31.85,40.12,39.84,45.52])
ax1.plot(x, y, 'rs-')
ax1.set_ylabel("Test Accuracy on PGD-Attacked CIFAR-100", color='red', fontsize=8)
ax1.tick_params(axis="y", labelcolor='red')



plt.savefig('results/toplogy_comparison_pdg_cifar100.png')