import os
import numpy as np
from matplotlib import pyplot as plt
import librosa.display
import seaborn as sns

import matplotlib

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 30}

matplotlib.rc('font', **font)

path = '/home/daripete/icassp2023/results/distribution'
out_path = '/home/daripete/icassp2023/plots'
file = 'code_values_main bottleneck quant6.npy'
band = 'cb'

fig, axs = plt.subplots(figsize=(18,7), nrows=1, ncols=1, sharex=True, sharey=True)
c = sns.color_palette("viridis", n_colors=100)

d1 = np.load(os.path.join(path,'p_40',file)).flatten()[:]
d2 = np.load(os.path.join(path,'bl1_40',file)).flatten()[:]
d3 = np.load(os.path.join(path,'bl2_40',file)).flatten()[:]

bins = np.load(os.path.join(path,'p_40','{}_bins.npy'.format(band)))
bins.sort()
sns.histplot(d1[(d1<=1.0) & (d1 >= -1)], bins=bins, kde=False, stat='probability', color=c[0], 
             ax=axs, line_kws={'linewidth':3}, alpha=0.5, label=r'$\mathbf{p}_{34:6}$')

bins = np.load(os.path.join(path,'bl1_40','{}_bins.npy'.format(band)))
bins.sort()
sns.histplot(d2[(d2<=1.0) & (d2 >= -1)], bins=bins, kde=False, stat='probability', color=c[30],
             ax=axs, line_kws={'linewidth':0.1}, alpha=0.5, label=r'$\mathbf{bl}_{1}$')

bins = np.load(os.path.join(path,'bl2_40','{}_bins.npy'.format(band)))
bins.sort()
sns.histplot(d3[(d3<=1.0) & (d3 >= -1)], bins=bins, kde=False, stat='probability', color=c[60],
             ax=axs, line_kws={'linewidth':0.1}, alpha=0.5, label=r'$\mathbf{bl}_{2}$')

axs.set_xticklabels([r'$\sim 34kbps$',r'$\sim 27kbps$',r'$\sim 20kbps$'], rotation = 0, ha="center")
axs.set_xticks([-0.9, 0.0, 0.9])
axs.legend(bbox_to_anchor=(0.6, 1.))
axs.grid()
plt.savefig(os.path.join(out_path,'cb_distribution.png'), bbox_inches='tight')

# %%
file = 'code_values_layer skip 3.npy'
band = 'hb'

fig, axs = plt.subplots(figsize=(18,7), nrows=1, ncols=1, sharex=True, sharey=True)
c = sns.color_palette("viridis", n_colors=100)

d1 = np.load(os.path.join(path,'p_40',file)).flatten()[:]
d2 = np.load(os.path.join(path,'bl1_40',file)).flatten()[:]
d3 = np.load(os.path.join(path,'bl2_40',file)).flatten()[:]

bins = np.load(os.path.join(path,'p_40','{}_bins.npy'.format(band)))
bins.sort()
sns.histplot(d1[(d1<=1.0) & (d1 >= -1)], bins=bins, kde=False, stat='probability', color=c[0], 
             ax=axs, line_kws={'linewidth':3}, alpha=0.5, label=r'$\mathbf{p}_{34:6}$')

bins = np.load(os.path.join(path,'bl1_40','{}_bins.npy'.format(band)))
bins.sort()
sns.histplot(d2[(d2<=1.0) & (d2 >= -1)], bins=bins, kde=False, stat='probability', color=c[30],
             ax=axs, line_kws={'linewidth':0.1}, alpha=0.5, label=r'$\mathbf{bl}_{1}$')

bins = np.load(os.path.join(path,'bl2_40','{}_bins.npy'.format(band)))
bins.sort()
sns.histplot(d3[(d3<=1.0) & (d3 >= -1)], bins=bins, kde=False, stat='probability', color=c[60],
             ax=axs, line_kws={'linewidth':0.1}, alpha=0.5, label=r'$\mathbf{bl}_{2}$')

axs.set_xticklabels([r'$\sim 6kbps$',r'$\sim 13kbps$',r'$\sim 20kbps$'], rotation = 0, ha="center")
axs.set_xticks([-0.9, 0.1, 0.9])
axs.legend(bbox_to_anchor=(0.6, 1.))
axs.grid()
plt.savefig(os.path.join(out_path,'hb_distribution.png'), bbox_inches='tight')