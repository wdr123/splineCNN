import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

plt.rcParams['font.size'] = 12
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['font.style'] = 'normal'

x_rob = np.array([4,6,8, ])
x_tar = np.array([10,12,14])
depth_rob = np.array([23,23,11, ])
depth_tar = np.array([20,18,8, ])

width_rob = np.array([10,10,4, ])
width_tar = np.array([6,6,2, ])

fig, ax = plt.subplots()

patterns = [ "/" , "x", "o", "*" ]

bar1 = ax.bar(x_rob, depth_rob, color='blue', label='A3_depth', hatch=patterns[0])
bar3 = ax.bar(x_rob, -width_rob, color='orange', label='A3_width', hatch=patterns[1])
bar2 = ax.bar(x_tar, depth_tar, color='red', label='RL_depth', hatch=patterns[2])
bar4 = ax.bar(x_tar, -width_tar, color='orange', label='RL_width', hatch=patterns[3])
ax.yaxis.set_major_formatter(lambda x, pos: f'{abs(x):g}')
ax.margins(x=0.1)
ax.set_xticks([4,6,8,10,12,14])
ax.set_xticklabels(['stage_1','stage_2','stage_3','stage_1','stage_2','stage_3',])
ax.set_xlabel('FLOPs')
ax.set_yticks([-30,-20,0,20,30])

for idx, rect in enumerate(bar1 + bar2):
    height = rect.get_height()
    if idx<3:
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, 'R', color='b', ha='center', va='bottom')
    elif idx==5:
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, 'R', color='red', ha='center', va='bottom')
    elif idx==4:
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, 'R', color='red', ha='center', va='bottom')
    else:
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, 'N', color='black', ha='center', va='bottom')

for idx, rect in enumerate(bar3 + bar4):
    height = rect.get_height()
    if idx==0:
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, 'C=0.45', color='black', ha='center', va='top')
    elif idx==1:
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, 'C=0.44', color='black', ha='center', va='top')
    elif idx==2:
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, 'C=0.56', color='black', ha='center', va='top')
    elif idx==3:
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, 'C=0.49', color='black', ha='center', va='top')
    elif idx==4:
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, 'C=0.52', color='black', ha='center', va='top')
    else:
        plt.text(rect.get_x() + rect.get_width() / 2.0, height, 'C=0.67', color='black', ha='center', va='top')

ax.set_ylabel(["$W_{i}, i@[1,2,3]$", '$D_{i}, i@[1,2,3]$'])
ax.legend(loc='lower right')



ax1 = ax.twinx()
x = np.array([6,12])
y = np.array([63.54,71.89])
ax1.plot(x, y, 'rs-')

text_width = 0.2
x1 = x + text_width
y1 = y
ax1.text(x1[0], y1[0], 'Lipschitz=3.4', color="blue")
ax1.text(x1[1], y1[1], 'Lipschitz=2.3', color="blue")
ax1.set_ylabel("Test Accuracy on AutoAttack CIFAR-10", color='red')
ax1.tick_params(axis="y", labelcolor='red')



plt.savefig('results/toplogy_autoattack_cifar10_20g.eps')
plt.savefig('results/toplogy_autoattack_cifar10_20g.png')