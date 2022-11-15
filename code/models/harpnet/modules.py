from torch.nn import BatchNorm1d, Conv1d
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
import os
from pathlib import Path
import math
import random
from torch.nn.utils.spectral_norm import spectral_norm
import torchaudio

from utils import AverageMeter, get_entropy
from loguru import logger

class Upsample(nn.Module):
    def __init__(self, in_f, out_f, kernel, scale_factor=2):
        super(Upsample, self).__init__()
        
        if scale_factor > 1:
            self.block = nn.Sequential(
                        nn.Conv1d(in_channels=in_f,
                                out_channels=out_f,
                                kernel_size=kernel,
                                padding=(kernel//2),
                                stride=1, dilation=1),
                        nn.Upsample(scale_factor=scale_factor, mode='nearest'),
                        nn.Conv1d(out_f,out_f, kernel_size=kernel, padding=kernel//2, stride=1),
                        BatchNorm1d(out_f),
                )
        else:
            self.block = nn.Sequential(
                        nn.Conv1d(in_channels=in_f,
                                out_channels=out_f,
                                kernel_size=kernel,
                                padding=(kernel//2),
                                stride=1, dilation=1),
                        BatchNorm1d(out_f),
            )
    def forward(self, x):
        return self.block(x)

class SNREntropyCheckpoint(pl.Callback):
    def __init__(self, monitor_delay, monitor_entropy, dirpath, overall_entropy=True):
        
        self.monitor_delay = monitor_delay
        self.best_loss = np.inf
        self.monitor_entropy = monitor_entropy
        self.snrs = []
        self.previous_ckpt_path = ""
        self.dirpath = dirpath
        self.overall_entropy = overall_entropy
        Path(self.dirpath).mkdir(parents=True, exist_ok=True)
        
    def _reset_batch_loss(self):
        self.snrs = []
        
    def _delete_previous_ckpt(self):
        if os.path.exists(self.previous_ckpt_path):
            os.remove(self.previous_ckpt_path)
        
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.snrs.append(outputs['snrs'])
        
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        
        if trainer.current_epoch > self.monitor_delay:
            
            est_bitrates = trainer.model.get_overall_entropy_avg()[1]
            target_bitrates = np.array([skip.target_bitrate for skip in trainer.model.skip_encoders])
            if self.overall_entropy:
                est_bitrates, target_bitrates = np.sum(est_bitrates,keepdims=True), np.sum(target_bitrates,keepdims=True)         
            f = 1500#target_bitrates // 100
            if  (
                (np.array([e > t-f for e, t in zip(est_bitrates,target_bitrates)]).all() and
                np.array([e < t+f for e, t in zip(est_bitrates,target_bitrates)]).all()) or
                (not self.monitor_entropy)
                ):
                val_loss_snr = torch.tensor(self.snrs).mean().item()
                if val_loss_snr < self.best_loss:
                    self.best_loss = np.round(val_loss_snr, 4)
                    filename = f"{trainer.current_epoch=}_{str(est_bitrates)=}_{self.best_loss=}.ckpt"
                    ckpt_path = os.path.join(self.dirpath,filename)
                    trainer.save_checkpoint(ckpt_path)
                    logger.info('New Best SNR:{} - Saving Model Checkpoint to: {}'.format(self.best_loss, ckpt_path))
                    #self._delete_previous_ckpt()
                    self.previous_ckpt_path = ckpt_path
                    
        trainer.model.entropy_control_update()
        self._reset_batch_loss()


class SkipEncoding(nn.Module):
    def __init__(self,         
        num_filters,
        target_bitrate=0,
        target_entropy=0,
        entropy_fuzz=0,
        H = 16384, # input samples
        kernel_size_down = 15,
        kernel_size_up = 15,
        quant_bins = None,
        quant_alpha = -10.0,
        quant_active = False,
        module_name='',
        stride = 1
    ):
        super(SkipEncoding, self).__init__()

        self.num_filters   = num_filters
        self.enc_conv     = nn.ModuleList()
        self.dec_conv     = nn.ModuleList()
        self.bn_enc       = nn.ModuleList()
        self.bn_dec       = nn.ModuleList()
        self.dec_num_filt = []
        self.H            = H
        self.kernel_size_down = kernel_size_down
        self.kernel_size_up   = kernel_size_up
        self.module_name = module_name

        self.leaky = nn.LeakyReLU(negative_slope=0.2)
        self.tanh = nn.Tanh()
        self.bottleneck_dims = (1,self.H)

        # Quant
        self.quant = None
        self.quant_bins = quant_bins
        self.quant_alpha = quant_alpha
        self.quant_active = quant_active
        self.target_bitrate = target_bitrate
        self.target_entropy = target_entropy
        self.entropy_fuzz = entropy_fuzz

        # Encoding Path
        self.enc_conv.append(nn.Sequential(
            nn.Conv1d(in_channels=num_filters[0], out_channels=num_filters[1], kernel_size=self.kernel_size_down, padding=(self.kernel_size_down // 2), stride=1),
            BatchNorm1d(num_filters[1])
        ))
        self.enc_conv.append(nn.Sequential(
            nn.Conv1d(in_channels=num_filters[1], out_channels=1, kernel_size=self.kernel_size_down, padding=(self.kernel_size_down // 2), stride=stride),
        ))

        self.quant = ScalarSoftmaxQuantization(
            bins = self.quant_bins,
            alpha = self.quant_alpha,
            code_length = self.bottleneck_dims[1],
            num_kmean_kernels = self.quant_bins.shape[0],
            feat_maps = self.bottleneck_dims[0],
            module_name = module_name
        )

        # Decoding Path
        self.dec_conv.append(Upsample(in_f=1, out_f=num_filters[1], kernel=self.kernel_size_down, scale_factor=stride))
        self.dec_conv.append(Upsample(in_f=num_filters[1], out_f=num_filters[0], kernel=self.kernel_size_down, scale_factor=1))
            
    def forward_skip(self, x):
        weighted_code_entropy = torch.zeros(1,2,dtype=torch.float)
        quant_loss = torch.zeros(1,dtype=torch.float)

        x = self.enc_conv[0](x)
        x = self.enc_conv[1](x)
        x = self.tanh(x)
        if self.quant_active:
            x, code_entropy, quant_loss = self.quant.forward_q(x)
            weighted_code_entropy = code_entropy
        #x = self.leaky(x)

        x = self.dec_conv[0](x)
        x = self.dec_conv[1](x)

        return x, weighted_code_entropy, quant_loss

class ScalarSoftmaxQuantization(nn.Module):
    def __init__(self, 
        bins,        
        alpha,
        code_length,
        num_kmean_kernels,
        feat_maps,
        module_name=''
        ):
        
        super(ScalarSoftmaxQuantization, self).__init__()

        self.eps = 1e-20
        #p = '/home/daripete/icassp2023/clusters_cb.npy' if 'main' in module_name else '/home/daripete/icassp2023/clusters_hb.npy'
        self.bins = bins
        #self.bins2 = torch.nn.Parameter(torch.from_numpy(np.load(p,allow_pickle=True)).squeeze())
        self.alpha = alpha
        self.code_length = code_length
        self.feat_maps   = feat_maps
        self.num_kmean_kernels = num_kmean_kernels

        # Entropy control
        self.code_entropy = 0
        self.tau  = 0.1
        self.tau2 = 1.0
        self.entropy_avg = AverageMeter()
        self.module_name = module_name
        
        self.acc = np.array([])
    
    def forward_q(self, x):

        '''
        x = [batch_size, feature_maps, floating_code] // [-1, 1, H]
        bins = [quant_num_bins] // [4]
        output = [-1, 1, H]
        '''

        input_size = x.size()
        weighted_code_entropy = torch.zeros(1,2,dtype=torch.float)
        weighted_quant_loss   = torch.zeros(1,2,dtype=torch.float)

        x = torch.unsqueeze(x, len(x.size()))
        floating_code = x.expand(input_size[0],self.feat_maps,self.code_length,self.num_kmean_kernels)

        bins_expand = torch.reshape(self.bins, (1, 1, 1, -1))
        dist = torch.abs(floating_code - bins_expand)
        soft_assignment = nn.Softmax(-1)(torch.divide((dist*-1), (1.0/self.alpha)))
        max_prob_bin = torch.topk(soft_assignment,1).indices

        if not self.training:
            hard_assignment = torch.reshape(F.one_hot(max_prob_bin, self.num_kmean_kernels),
                                        (input_size[0], self.feat_maps, self.code_length, self.num_kmean_kernels))
            hard_assignment = hard_assignment.type(torch.float)
            assignment = hard_assignment
        else:
            assignment = soft_assignment

        # Quantization regularization term, prevent alpha from getting to big
        weighted_quant_loss[:,0] = torch.sum(torch.mean(torch.sqrt(assignment+self.eps),(0,1,2))) - 1.0
        weighted_quant_loss[:,1] = self.tau2
        p = torch.mean(assignment, dim=(0,1,2))
        # Compute entropy loss
        self.code_entropy = get_entropy(p+self.eps)
        self.entropy_avg.update(self.code_entropy.detach().item())
        
        # Weighted entropy regularization term
        weighted_code_entropy[:,0] = self.code_entropy
        weighted_code_entropy[:,1] = self.tau
                        
        bit_code = torch.matmul(assignment,self.bins)
        np_bc = bit_code.clone().squeeze().detach().cpu().numpy()
        # if len(self.acc) == 0:
        #     self.acc = np_bc
        # else:
        #     self.acc = np.concatenate([self.acc, np_bc])
        #import pdb; pdb.set_trace()
        return bit_code, weighted_code_entropy, weighted_quant_loss