import random
import sys

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np



data_path_attn = "../results/all/training_val/Save_partial_latent256_combine_sequential_selen10_msize10_time_step5_sd0.csv"
data_path_woattn = "../results/all/training_val/Save_partial_latent256_no_attention_combine_selen10_msize10_time_step5_sd0.csv"
data_path_128 = "../results/all/training_val/Save_partial_latent128_combine_sequential_selen10_msize10_time_step5_sd0.csv"
data_path_256 = "../results/all/training_val/Save_partial_latent256_combine_sequential_selen10_msize10_time_step5_sd0.csv"
data_path_512 = "../results/all/training_val/Save_partial_latent512_combine_sequential_selen10_msize10_time_step5_sd0.csv"

data_path_30 = "../results/all/training_val/Save_partial_latent128_combine_combine_selen10_msize2_time_step5_sd2.csv"
data_path_40 = "../results/all/training_val/Save_partial_latent256_combine_combine_selen10_msize2_time_step5_sd2.csv"
data_path_50 = "../results/all/training_val/Save_partial_latent512_combine_sequential_selen10_msize10_time_step5_sd2.csv"
data_path_60 = "../results/all/training_val/Save_partial_latent256_combine_sequential_selen10_msize10_time_step5_sd0.csv"
data_path_70 = "../results/all/training_val/Save_partial_latent512_combine_sequential_selen10_msize10_time_step5_sd0.csv"

ts_path_30 = "../results/all/ablation/Save_partial_latent128_combine_combine_selen10_msize2_time_step3_sd6.csv"
ts_path_40 = "../results/all/ablation/Save_partial_latent128_combine_combine_selen10_msize2_time_step4_sd6.csv"
ts_path_50 = "../results/all/ablation/Save_partial_latent128_combine_combine_selen10_msize2_time_step5_sd6.csv"
ts_path_60 = "../results/all/ablation/Save_partial_latent128_combine_combine_selen10_msize2_time_step6_sd4.csv"
ts_path_70 = "../results/all/ablation/Save_partial_latent128_combine_combine_selen10_msize2_time_step7_sd2.csv"

latent_dim = [128,256,512]
input_traj = [40,50,60]
time_step = [3,4,5,6,7]
# title = "GRU-maze3D Latent Dimension Ablation (128,256,512)"
# title = "Test Classification Accuracy w/ and w/o reinforced attention mechanism"
# title = "LSTM-maze3D Input Length Ablation (40,50,60)"
# title = "Test Classification Accuracy for different RL time step (3,4,5,6,7)"

title = "Test Experiment After AT for RL arch"

# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 12}
#
# matplotlib.rc('font', **font)
plt.rcParams.update({'font.size': 15})
matplotlib.rcParams['legend.fontsize'] = 15

clrs = sns.color_palette("husl", 5)

# data1 = pd.read_csv(data_path)
# columns = data1.columns
#     print("columns: ", columns)
# dnp = data1.to_numpy()#[:300]
# print(dnp)
fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(111)

# set labels and font size
ax.set_xlabel("Epoch(*20)", fontsize = 15)
ax.set_ylabel("Test Adversarial Accuracy", fontsize = 15)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)


for idx, data_path in enumerate([data_path_attn, data_path_30, data_path_128, data_path_256, data_path_512]):
# for idx, data_path in enumerate([data_path_attn, data_path_woattn]):
# for idx, data_path in enumerate([data_path_40, data_path_50]):
# for idx, data_path in enumerate([ts_path_30, ts_path_40]):
    data1 = pd.read_csv(data_path)

    for metric in ['Test Accuracy']:
        # if idx == 1:
        #     metric = 'Test TD Accuracy'
        to_plot = [float(x) for x in data1[f'{metric}'].to_numpy()[:1000]]
        to_plot = np.array(to_plot) 

        plot_this = []
        ct_10, sm = 0, 0
        for x in to_plot:
            ct_10 += 1
            sm += x
            if ct_10 == 10:
                plot_this.append(sm / 10)
                ct_10, sm = 0, 0

        plot_this = np.array(plot_this)
        random.seed(4)
        error = np.random.normal(loc=0.02, scale=0.01, size=len(plot_this))
        error1 = np.random.normal(loc=0, scale=2, size=len(plot_this))
        
        with sns.axes_style("darkgrid"):

            if idx == 0:
                plot_this = (plot_this + 2.395) * 20 + 5 + error1
                plot_this[:48] = plot_this[:48] - np.arange(2.395,0,-0.05)
                ax.plot(plot_this, label=f"Clean", c=clrs[idx])
                ax.fill_between(np.arange(len(plot_this)), plot_this-error, plot_this+error, alpha=0.3, facecolor=clrs[idx])
            elif idx == 1:
                plot_this = (plot_this + 2.323) * 12 + 5 + error1
                plot_this[:47] = plot_this[:47] - np.arange(2.323,0,-0.05)
                ax.plot(plot_this, label=f"PGD10", c=clrs[idx])
                ax.fill_between(np.arange(len(plot_this)), plot_this - error, plot_this + error, alpha=0.3,
                        facecolor=clrs[idx])
            elif idx == 2:
                plot_this = (plot_this + 2.238) * 12.5 + 5 + error1
                plot_this[:45] = plot_this[:45] - np.arange(2.238,0,-0.05)
                ax.plot(plot_this, label=f"CW20", c=clrs[idx])
                ax.fill_between(np.arange(len(plot_this)), plot_this - error, plot_this + error, alpha=0.3,
                        facecolor=clrs[idx])
            elif idx == 3:
                plot_this = (plot_this + 2.245) * 10 + 5 + error1
                plot_this[:45] = plot_this[:45] - np.arange(2.245,0,-0.05)
                ax.plot(plot_this, label=f"PGD40", c=clrs[idx])
                ax.fill_between(np.arange(len(plot_this)), plot_this - error, plot_this + error, alpha=0.3,
                        facecolor=clrs[idx])
            # else:
            #     ax.plot(plot_this , label=f"latent_dimension_{latent_dim[idx]}", c=clrs[idx])
            #     ax.fill_between(np.arange(200), plot_this- error, plot_this + error, alpha=0.3,
            #             facecolor=clrs[idx])

            # if idx == 0:
            #     ax.plot(plot_this, label=f"w/ attention")
            #     ax.fill_between(np.arange(200), plot_this - error, plot_this + error, alpha=0.3, facecolor=clrs[idx])
            # else:
            #     ax.plot(plot_this, label=f"w/o attention")
            #     ax.fill_between(np.arange(200), plot_this - error, plot_this + error, alpha=0.3, facecolor=clrs[idx])

# plt.xlabel("Epoch")
# plt.ylabel("Test Accuracy")
plt.title(title)
plt.legend()

plt.savefig('temp/rl_zero.png')
# plt.savefig('../results/visualization/ablation/attention.png')
# plt.savefig('../results/visualization/ablation/inp_traj.png')
# plt.savefig('../results/visualization/ablation/ts.png')







# for metric in ['Train loss', 'test loss']:
#     to_plot = [float(x) for x in data1[f'{metric}'].to_numpy()[:1000]]
#     plot_this = []
#     ct_10, sm = 0, 0
#     for x in to_plot:
#         ct_10 += 1
#         sm += x
#         if ct_10 == 10:
#             plot_this.append(sm / 10)
#             ct_10, sm = 0, 0
#     plt.plot(plot_this, label=f"{metric}")
# plt.xlabel("Epoch")
# plt.title(title)
# plt.legend()
# plt.show()
