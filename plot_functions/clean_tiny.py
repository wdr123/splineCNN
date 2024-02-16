import matplotlib.pyplot as plt
import numpy as np

x_rob = np.array([4,5,6,  ])
x_car = np.array([10,11,12 ])
depth_rob = np.array([17,17,8, ])*1.7*1.7
depth_car = np.array([15,15,10])*1.5*1.8

width_rob = np.array([8,9,4, ])*1.5*1.6
width_car = np.array([6,6,6])*1.1*1.2

fig, ax = plt.subplots()
ax1 = ax.twinx()

ax.bar(x_rob, depth_rob, color='blue', label='rob')
ax.bar(x_rob, -width_rob, color='orange', label='rob')
ax.bar(x_car, depth_car, color='red', label='car')
ax.bar(x_car, -width_car, color='orange', label='car')
ax.yaxis.set_major_formatter(lambda x, pos: f'{abs(x):g}')
ax.margins(x=0.1)
ax.set_xticks([5,11,])
ax.set_xticklabels(['rob_40G','car_40G'])
ax.set_xlabel('FLOPs')
ax.set_yticks([-40,-20,0,20,40])
ax.set_ylabel(['W_i','D_i'])

x = np.array([5,11,])
y = np.array([68.29,73.15])
ax1.plot(x, y, 'rs-')
ax1.set_ylabel("Test Accuracy on Clean Tiny-ImageNet", color='red', fontsize=8)
ax1.tick_params(axis="y", labelcolor='red')



plt.savefig('results/toplogy_comparison_clean_tiny.png')