#%%
import os, sys
import torch
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
import soundfile as sf
import librosa
import librosa.display

import matplotlib

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 30}

matplotlib.rc('font', **font)

path = '/home/daripete/icassp2023/results/ablation_audio'
out_path = '/home/daripete/icassp2023/plots'

s_frame, e_frame = 1125, 1300

bl1_ab = sf.read(os.path.join(path,'bl1_ab','0_pred_hb.wav'))[0]
bl1_noab = sf.read(os.path.join(path,'bl1_noab','0_pred_hb.wav'))[0]

p_ab = sf.read(os.path.join(path,'p_ab','0_pred_hb.wav'))[0]
p_noab = sf.read(os.path.join(path,'p_noab','0_pred_hb.wav'))[0]

input = sf.read(os.path.join(path,'input','0_input_hb.wav'))[0]

S_bl1_ab = np.abs(librosa.stft(bl1_ab))
S_bl1_noab = np.abs(librosa.stft(bl1_noab))
S_p_ab = np.abs(librosa.stft(p_ab))
S_p_noab = np.abs(librosa.stft(p_noab))
S_input = np.abs(librosa.stft(input))


fig, axs = plt.subplots(figsize=(7.85,3.04), nrows=1, ncols=1, sharex=True, sharey=True)
librosa.display.specshow(librosa.amplitude_to_db(S_input,ref=np.max)[:,s_frame:e_frame], y_axis='hz', ax=axs, sr=32000, cmap='magma')
axs.set_ylabel('kHz')
axs.set_yticks([10000,16000])
axs.set_yticklabels(['10','16'])
axs.set_ylim([8000,16000])

fig.savefig(os.path.join(out_path,'ab_input.pdf'), bbox_inches='tight')

fig1, axs1 = plt.subplots(figsize=(7.85,3.04), nrows=1, ncols=1, sharey=True)
fig2, axs2 = plt.subplots(figsize=(7.85,3.04), nrows=1, ncols=1, sharey=True)
librosa.display.specshow(librosa.amplitude_to_db(S_bl1_ab,ref=np.max)[:,s_frame:e_frame], y_axis='hz', ax=axs1, sr=32000, cmap='magma')
librosa.display.specshow(librosa.amplitude_to_db(S_bl1_noab,ref=np.max)[:,s_frame:e_frame], y_axis='hz', ax=axs2, sr=32000, cmap='magma')
# axs[0].set_ylabel('kHz')
# axs[1].set_ylabel('kHz')

axs1.set_yticks([0])
axs1.set_yticklabels([''])
axs2.set_yticks([0])
axs2.set_yticklabels([''])
axs1.set_ylabel('')
axs2.set_ylabel('')
axs1.set_ylim([8000,16000])
axs2.set_ylim([8000,16000])

fig.subplots_adjust(hspace=0.3)
fig1.savefig(os.path.join(out_path,'ab_bl1_0.pdf'), bbox_inches='tight')
fig2.savefig(os.path.join(out_path,'ab_bl1_cb.pdf'), bbox_inches='tight')

fig1, axs1 = plt.subplots(figsize=(7.85,3.04), nrows=1, ncols=1, sharey=True)
fig2, axs2 = plt.subplots(figsize=(7.85,3.04), nrows=1, ncols=1, sharey=True)
librosa.display.specshow(librosa.amplitude_to_db(S_p_ab,ref=np.max)[:,s_frame:e_frame], y_axis='hz', ax=axs1, sr=32000, cmap='magma')
librosa.display.specshow(librosa.amplitude_to_db(S_p_noab,ref=np.max)[:,s_frame:e_frame], y_axis='hz' , ax=axs2, sr=32000, cmap='magma')
# axs[0].set_ylabel('kHz')
# axs[1].set_ylabel('kHz')

axs1.set_yticks([0])
axs1.set_yticklabels([''])
axs2.set_yticks([0])
axs2.set_yticklabels([''])
axs1.set_ylabel('')
axs2.set_ylabel('')
axs1.set_ylim([8000,16000])
axs2.set_ylim([8000,16000])
# axs[1].set_xticks([50,150,250,350])
# axs[1].set_xticklabels(['50','150','250','350'])

fig.subplots_adjust(hspace=0.3)
fig1.savefig(os.path.join(out_path,'ab_p_0.pdf'), bbox_inches='tight')
fig2.savefig(os.path.join(out_path,'ab_p_cb.pdf'), bbox_inches='tight')
# %%
