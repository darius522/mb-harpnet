#%%
import pandas as pd
import numpy as np
import os
import csv
from matplotlib import pyplot as plt
from glob import glob

import random
import matplotlib

import json
import ast
import seaborn as sns

r_dir = '../raw_results'


pretty_names = {'C1':r"$\mathbf{bl1}$-$40kbps$",
                'C2':r"$\mathbf{bl2}$-$40kbps$",
                'C3':r"$\mathbf{p}_{34:6}-40kbps$",
                'C4':r"$\mathbf{bl1}$-$48kbps$",
                'C5':r"$\mathbf{bl2}$-$48kbps$",
                'C6':r"$\mathbf{p}_{41:7}-48kbps$",
                'C7':r"$\mathbf{HE}$-$\mathbf{AAC}$-$40kbps$",
                'reference':r"Hidden Ref.",
                'anchor35':r"Anchor - 3.5kHz"}

def plotBoxMetrics(results, plotTitle=''):

	font = {'family' : 'normal',
		'size'   : 30}
	matplotlib.rc('font', **font)

	matplotlib.rcParams['pdf.fonttype'] = 42
	matplotlib.rcParams['ps.fonttype'] = 42
	matplotlib.rcParams['axes.unicode_minus'] = False
	# matplotlib.rcParams['text.usetex'] = True
	# #matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command

	f = plt.figure(figsize=(18,9))
	names = ['reference','C7','C6','C5','C4','C3','C2','C1','anchor35']
	xs     = []
	vals   = []
	for i, name in enumerate(names):
		vals.append(np.asarray(results[name],dtype=int))
		xs.append(np.random.normal(i + 1, 0.04, len(results[name])))

	#convert names
	real_names = [pretty_names[i] for i in names]

	sns.set_style("whitegrid")  # "white","dark","darkgrid","ticks"
	# p1 = np.array(sns.color_palette("Reds", 3))
	# p2 = np.array(sns.color_palette("Blues", 3))
	# palette = np.concatenate([p1,p2])
	boxprops = dict(linestyle='-', linewidth=2.5, color='#00145A', facecolor='white')
	flierprops = dict(marker='o', markersize=0,
					  linestyle='none')
	whiskerprops = dict(color='#00145A')
	capprops = dict(color='#00145A')
	medianprops = dict(linewidth=2.0, linestyle='-', color='#6e0101')
	meanprops = dict(linewidth=2.0, linestyle='--', color='#017507')
	bb = plt.boxplot(vals, labels=real_names, labelcolors='black,'
		notch=False, 
		boxprops=boxprops, 
		whiskerprops=whiskerprops,
		capprops=capprops, 
		flierprops=flierprops, 
		medianprops=medianprops,
		showmeans=True,
		meanline=True,
		meanprops=meanprops,
		showfliers=True,
  		patch_artist=True, 
    	zorder=1) 
 
	p1 = np.array(sns.color_palette("Greens_r", 10))[2:5,...]
	p2 = np.array(sns.color_palette("Blues_r", 10))[2:5,...]
	palette = np.concatenate([p1,p2])
	alphas = (np.ones(palette.shape[0])*0.7).reshape(-1,1)
	palette = np.append(palette,alphas,axis=1)
	for patch, color in zip(bb['boxes'][2:-1], palette):
		patch.set_facecolor(color)
		patch.get_text()

	for i, (x, val) in enumerate(zip(xs, vals)):
		low = np.quantile(val,0.25)
		high  = np.quantile(val,0.75)
		val_n = val[(val > low) & (val < high)]
		x_n = x[(val > low) & (val < high)]
		plt.scatter(x_n, val_n, alpha=0.7, color='black', zorder=10000)

	#plt.show()
	plt.ylabel('Subjective Score')
	plt.xticks(np.arange(len(real_names))+1, real_names, rotation=45)
	plt.tight_layout()
	plt.savefig('../plots/mushra.pdf')
	plt.savefig('../plots/mushra.png')

def plot_models_mse():

	font = {'family' : 'normal',
		'size'   : 30}
	matplotlib.rc('font', **font)
	matplotlib.rcParams['pdf.fonttype'] = 42
	matplotlib.rcParams['ps.fonttype'] = 42

	results = glob(os.path.join(r_dir,'*.npy'))
	bl = sorted([x for x in results if 'bl' in x])
	hl = sorted([x for x in results if 'hl' in x])
	bs = sorted([x for x in results if 'bs' in x])
	hs = sorted([x for x in results if 'hs' in x])

	bl_means = []
	bl_stds  = []
	hl_means = []
	hl_stds  = []
	bs_means = []
	bs_stds  = []
	hs_means = []
	hs_stds  = []

	for i in range(4):
		bl_means.append(np.mean(np.load(bl[i])))
		bl_stds.append(np.std(np.load(bl[i])))

		hl_means.append(np.mean(np.load(hl[i])))
		hl_stds.append(np.std(np.load(hl[i])))

		bs_means.append(np.mean(np.load(bs[i])))
		bs_stds.append(np.std(np.load(bs[i])))

		hs_means.append(np.mean(np.load(hs[i])))
		hs_stds.append(np.std(np.load(hs[i])))


	fig, ax = plt.subplots(figsize=(8,8))

	e1 = plt.errorbar([1,2,3,4], bs_means, bs_stds, marker='o', barsabove=True, capthick=2.0, capsize=10.0, color='tab:red', label='Baseline 40-kbps', alpha=0.8)
	e1[-1][0].set_linestyle('--')
	e2 = plt.errorbar([1,2,3,4], hs_means, hs_stds, marker='o', barsabove=True, capthick=2.0, capsize=10.0, color='black', label='HARP-Net 40-kbps', alpha=0.8)
	e2[-1][0].set_linestyle('--')
	plt.legend(loc='lower center')
	plt.ylabel('SNR (dB)')
	plt.xticks([1, 2, 3, 4],['1 Skip', '2 Skips', '3 Skips', '4 Skips'])
	plt.ylim([-1,23])

	# e1 = ax[1].errorbar([1,2,3,4], bl_means, bl_stds, marker='o', barsabove=True, capthick=2.0, capsize=10.0, color='tab:red', label='Baseline 48-kbps', alpha=0.8)
	# e1[-1][0].set_linestyle('--')
	# e2 = ax[1].errorbar([1,2,3,4], hl_means, hl_stds, marker='o', barsabove=True, capthick=2.0, capsize=10.0, color='black', label='HARPNET 48-kbps', alpha=0.8)
	# e2[-1][0].set_linestyle('--')
	# ax[1].legend(loc='lower center')
	# ax[1].set_xticks([1, 2, 3, 4])
	# ax[1].set_xticklabels(['1 Skip', '2 Skips', '3 Skips', '4 Skips'])
	# ax[1].set_ylim([-1,23])

	plt.grid()
	plt.tight_layout()
	plt.savefig('../plots/mse_models_40.pdf')


def plot_mushra():
	stimuli = ['C1','C2','C3','C4','C5','C6','C7','reference','anchor35']
	results = {i:[] for i in stimuli}
	path = '../results/mushra.json'
	with open(path) as json_file:
		data = json.load(json_file)

	participant_ids = list(data.keys())
	participant_ids.remove('1665643257412')
	for p_id in participant_ids:
		all_trials = json.loads(data[p_id])['trials']
		missed_ref = 0
		for trials in all_trials: 
			for trial in trials['responses']:
				if trial['stimulus'] == 'reference' and trial['score'] != '100':
					missed_ref += 1
				results[trial['stimulus']].append(trial['score'])
		print(p_id, missed_ref)
	plotBoxMetrics(results)

#plot_models_mse()
plot_mushra()







# %%
