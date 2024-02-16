# importing package
import matplotlib.pyplot as plt
import numpy as np

# create data
plt.rcParams['font.size'] = 12
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['font.style'] = 'normal'

x = [0.5,1.3, 3.1,3.9]
x1 = [2.1, 4.7]

stage1 = np.array([9,5,16,5])
stage2 = np.array([7,5,17,5])
stage3 = np.array([7,5,9,3])

ratio1 = stage1 / (stage1+stage2+stage3)
ratio2 = stage2 / (stage1+stage2+stage3)
ratio3 = stage3 / (stage1+stage2+stage3)

total_depth_5g = stage1[0] + stage2[0] + stage3[0]
total_depth_10g = stage1[2] + stage2[2] + stage3[2]
total_width_5g = stage1[1] + stage2[1] + stage3[1]
total_width_10g = stage1[3] + stage2[3] + stage3[3]

ratio_depth = np.array([total_depth_5g/(total_depth_5g+total_width_5g), total_depth_10g/(total_depth_10g+total_width_10g)])
ratio_width = np.array([total_width_5g/(total_depth_5g+total_width_5g), total_width_10g/(total_depth_10g+total_width_10g)])
# plot bars in stack manner
fig, ax = plt.subplots()

ax.bar(x, ratio1, width=0.5, color='r')
ax.bar(x, ratio2, width=0.5, bottom=ratio1, color='b')
ax.bar(x, ratio3, width=0.5, bottom=ratio1+ratio2, color='y')
ax.bar(x1, ratio_depth, width=0.5, color='m')
ax.bar(x1, ratio_width, width=0.5, bottom=ratio_depth, color='c')

ax.set_xticks(np.concatenate([x,x1]))
ax.set_xticklabels(['5G_D','5G_W','10G_D','10G_W','5G_T','10G_T'])
ax.set_xlabel("Computation Budget (FLOPs)")
ax.set_ylabel("Distribution")
ax.set_ylim(0,1)



plt.legend(["Stage1", "Stage2", "Stage3", "Total Depth", "Total Width"])
plt.title("D/W/D&W Sum Ratio for RL selected Archs \n on auto attcked cifar-100")

plt.savefig('results/stack_hist_pgd_cifar100.png')
plt.savefig('results/stack_hist_pgd_cifar100.eps')
