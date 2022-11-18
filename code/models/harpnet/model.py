from torch.nn import BatchNorm1d
import torch
import torch.nn as nn
import torchaudio.transforms as T
import utils

from pytorch_lightning.core.lightning import LightningModule

import auraloss
from loguru import logger
from pytorch_lightning.utilities import rank_zero_only

from .modules import (
    SkipEncoding, 
    Upsample
)

import numpy as np
import soundfile as sf
import os

def powspace(start, stop, power, num):
    start = np.power(start, 1/float(power))
    stop = np.power(stop, 1/float(power))
    return np.power( np.linspace(start, stop, num=num), power)
        
class HARPNet(LightningModule):
    
    def __init__(self,
        H = 16384,
        n_ch = 1,
        kernel_size_down = 15,
        kernel_size_up = 15,
        quant_num_bins = 2**5,
        tau_changes = [],
        quant_alpha = -10,
        alpha_decrease = 0.0,
        alpha_delay = -1.0,
        entropy_redux_delay = -1.0,
        entropy_loss_weight_factor = 0.016,
        sr = 32000,
        target_bitrates = [48000,16000],
        bitrate_fuzzes = [480,160],
        output_audio = False,
        loss_weights={'snr':1.0,'entropy':1.0,'quant':0.0, 'freq':1.0},
        scaler = 1.0,
        baseline = True,
        lr = 0.001, patience = 20, lr_decay_patience = 5, lr_decay_gamma = 0.3, weight_decay = 0.00001, **kwargs
    ):

        super(HARPNet, self).__init__()
        self.save_hyperparameters()
        
        print(kwargs)
        for k, i in kwargs.items():
            logger.info('{} -> {}'.format(k, i))
        
        self.lr = lr
        self.sr = sr
        self.patience = patience
        self.lr_decay_patience = lr_decay_patience
        self.lr_decay_gamma = lr_decay_gamma
        self.weight_decay = weight_decay
        
        #Loss
        self.loss = self.loss_fn
        self.snr = auraloss.time.SNRLoss()
        self.freq = auraloss.freq.STFTLoss(w_lin_mag=0.0, w_log_mag=1.0, w_phs=0.0, w_sc=0.0, output='loss')
        self.loss_weights = loss_weights
        self.loss_weights['weighted_entropy'] = 0.0
        self.baseline = baseline
        
        self.enc_conv      = nn.ModuleList()
        self.dec_conv      = nn.ModuleList()
        self.bn_enc        = nn.ModuleList()
        self.bn_dec        = nn.ModuleList()
        self.skip_encoders = nn.ModuleList()
        self.skip          = []
        self.H             = H
        self.channel       = n_ch
        self.kernel_size_down = kernel_size_down
        self.kernel_size_up   = kernel_size_up
        self.bottleneck_dims  = []
        
        self.codes = []

        self.leaky = nn.LeakyReLU()
        self.tanh  = nn.Tanh()
        
        self.resampler = T.Resample(16000, sr, dtype=torch.float32)

        # Quant
        self.quant_active = True
        self.alpha_delay = alpha_delay
        self.entropy_redux_delay = entropy_redux_delay
        self.entropy_loss_weight_factor = entropy_loss_weight_factor
        self.quant_num_bins = quant_num_bins
        self.quant_alpha = torch.nn.Parameter(torch.tensor(quant_alpha, dtype=torch.float32), requires_grad=False)
        self.register_parameter(name='alpha', param=(self.quant_alpha))
        self.quant_bins_cb = torch.nn.Parameter(torch.linspace(-1,1,self.quant_num_bins, dtype=torch.float32), requires_grad=True)
        self.register_parameter(name='bins_cb', param=(self.quant_bins_cb))
        self.quant_bins_hb = torch.nn.Parameter(torch.linspace(-1,1,self.quant_num_bins, dtype=torch.float32), requires_grad=True)
        self.register_parameter(name='bins_hb', param=(self.quant_bins_hb))
        self.quant_losses = torch.zeros(1,dtype=torch.float)
        
        # Entropy
        self.tau_changes     = tau_changes
        self.code_entropies = torch.zeros(1,2,dtype=torch.float)
        self.code_bitrates  = torch.zeros(1,2,dtype=torch.float)
        self.quant_losses   = torch.zeros(1,2,dtype=torch.float)
        self.alpha_decrease = alpha_decrease#powspace(1, 1000.0, 5, 291)
        
        # Test
        self.output_audio = output_audio
        self.scaler = scaler
        
        ch = 50
        # Encoding Path
        # C x 16384
        self.enc_conv.append(nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=ch, kernel_size=self.kernel_size_down, padding=(self.kernel_size_down // 2), stride=1),
            BatchNorm1d(ch)
        ))
        # C x 8192
        self.enc_conv.append(nn.Sequential(
            nn.Conv1d(in_channels=ch, out_channels=ch, kernel_size=self.kernel_size_down, padding=(self.kernel_size_down // 2), stride=1),
            BatchNorm1d(ch)
        ))
        # C x 8192
        self.enc_conv.append(nn.Sequential(
            nn.Conv1d(in_channels=ch, out_channels=ch, kernel_size=self.kernel_size_down, padding=(self.kernel_size_down // 2), stride=1),
            BatchNorm1d(ch)
        ))
        # C x 8192
        self.enc_conv.append(nn.Sequential(
            nn.Conv1d(in_channels=ch, out_channels=ch, kernel_size=self.kernel_size_down, padding=(self.kernel_size_down // 2), stride=1),
            BatchNorm1d(ch)
        ))
        # C x 4096
        self.enc_conv.append(nn.Sequential(
            nn.Conv1d(in_channels=ch, out_channels=ch, kernel_size=self.kernel_size_down, padding=(self.kernel_size_down // 2), stride=1),
            BatchNorm1d(ch)
        ))
        # C x 2048
        self.enc_conv.append(nn.Sequential(
            nn.Conv1d(in_channels=ch, out_channels=ch, kernel_size=self.kernel_size_down, padding=(self.kernel_size_down // 2), stride=2),
        ))
          
        # Skips
        self.skip_encoders.append(
            SkipEncoding(num_filters=[ch,ch//2], H=8192,
                         quant_bins=self.quant_bins_cb, quant_alpha=self.quant_alpha, 
                         quant_active=self.quant_active, module_name='main bottleneck quant'+str(6)),
        )

        self.bottleneck_to_signal = nn.Sequential(
                                                  Upsample(in_f=ch, out_f=ch, kernel=self.kernel_size_down, scale_factor=1),
                                                  self.leaky,
                                                  Upsample(in_f=ch, out_f=ch, kernel=self.kernel_size_down, scale_factor=1),
                                                  self.leaky,
                                                  Upsample(in_f=ch, out_f=ch, kernel=self.kernel_size_down, scale_factor=1),
                                                  self.leaky,
                                                  Upsample(in_f=ch, out_f=ch, kernel=self.kernel_size_down, scale_factor=1),
                                                  self.leaky,
                                                  Upsample(in_f=ch, out_f=ch, kernel=self.kernel_size_down, scale_factor=1),
                                                  self.leaky,
                                                  nn.Conv1d(in_channels=ch, out_channels=1, kernel_size=1),
                                                  )
        self.skip_encoders.append(
            SkipEncoding(num_filters=[ch,ch//2], H=self.H, 
                         quant_bins=self.quant_bins_hb, quant_alpha=self.quant_alpha, 
                         quant_active=self.quant_active, module_name='layer skip '+str(3), stride=1)
        )

        self.bottleneck_dims = self.skip_encoders[0].bottleneck_dims
        print('Bottleneck shapes: ', [skip.bottleneck_dims for skip in self.skip_encoders])
        
        # Decoding Path
        # C x 2048
        self.dec_conv.append(Upsample(in_f=ch, out_f=ch, kernel=self.kernel_size_down, scale_factor=2))
        # C x 4096
        self.dec_conv.append(Upsample(in_f=ch, out_f=ch, kernel=self.kernel_size_down, scale_factor=1))
        # C x 8192
        self.dec_conv.append(Upsample(in_f=ch, out_f=ch, kernel=self.kernel_size_down, scale_factor=1))
        # C x 8192
        self.dec_conv.append(Upsample(in_f=ch*2, out_f=ch, kernel=self.kernel_size_down, scale_factor=1))
        # C x 8192
        self.dec_conv.append(Upsample(in_f=ch, out_f=ch, kernel=self.kernel_size_down, scale_factor=1))
        # C x 16384
        self.dec_conv.append(nn.Conv1d(in_channels=ch, out_channels=1, kernel_size=1))
        
        self.set_network_entropy_target(target_bitrates, bitrate_fuzzes, sr, H, 64)
        self.btlk_dims       = [skip.bottleneck_dims for skip in self.skip_encoders]
        self.target_ents     = [skip.target_entropy for skip in self.skip_encoders]
        self.target_bitrates    = [skip.target_bitrate for skip in self.skip_encoders]
        self.fuzzes          = [skip.entropy_fuzz for skip in self.skip_encoders]
        logger.info('Network with target entropy: {}, target bitrate: {}, bitrate fuzz: {}, bottleneck dims: {}'\
            .format(self.target_ents, self.target_bitrates, self.fuzzes, self.btlk_dims))
        
    def forward(self,x):
        self.skip = []
        self.code_entropies = torch.zeros(1,2,dtype=torch.float)
        self.code_bitrates  = torch.zeros(1,2,dtype=torch.float)
        self.quant_losses   = torch.zeros(1,2,dtype=torch.float)

        x    = self.enc_conv[0](x)
        x    = self.enc_conv[1](self.leaky(x))
        x_s1 = self.enc_conv[2](self.leaky(x))
        x    = self.enc_conv[3](self.leaky(x_s1.clone()))
        x    = self.enc_conv[4](self.leaky(x))
        x    = self.enc_conv[5](self.leaky(x))
        
        # Bottleneck / Skip
        y_cb, code_entropy, quant_loss = self.skip_encoders[0].forward_skip(x)
        skip_layer1, skip_entropy, skip_quant_loss = self.skip_encoders[-1].forward_skip(x_s1)
        if self.quant_active:
            code_br = utils.entropy_to_bitrate(code_entropy, 16000, self.H/2, 64, self.skip_encoders[0].bottleneck_dims)
            skip_br = utils.entropy_to_bitrate(skip_entropy, self.sr, self.H, 64, self.skip_encoders[1].bottleneck_dims)
            self.code_entropies = torch.cat((code_entropy,skip_entropy))
            self.code_bitrates  = torch.cat((code_br,skip_br))
            self.quant_losses = torch.cat((quant_loss,skip_quant_loss))
            
        x = self.dec_conv[0](y_cb.clone())
        x = self.dec_conv[1](self.leaky(x))
        x = self.dec_conv[2](self.leaky(x))
        x = self.dec_conv[3](self.leaky(torch.cat((x, skip_layer1), 1)))
        x = self.dec_conv[4](self.leaky(x))
        y_hb = self.dec_conv[5](self.leaky(x))

        return y_hb, self.bottleneck_to_signal(y_cb)

    def entropy_control_update(self):
        '''
        check soft assignment's entropy for each quantizer module.
        adjust quantizer lambda according to target entropy
        '''
        if self.quant_active and self.current_epoch > self.alpha_delay:
            d = self.alpha_decrease #if self.current_epoch < 150 else 1000.0
            self.quant_alpha += d
        if self.quant_active and self.current_epoch > self.entropy_redux_delay:
            # entropy control weight
            new_weight = self.entropy_loss_weight_factor * (self.current_epoch-self.entropy_redux_delay)
            if new_weight < 1.0:
                self.loss_weights['weighted_entropy'] = new_weight * self.loss_weights['entropy']
            self.reset_entropy_hists()

    def entropy_loss(self):
        '''
        combine all quantizer modules' soft-assignment entropy mean
        '''
        targets_bitrates = torch.tensor([skip.target_bitrate for skip in self.skip_encoders]) * self.loss_weights['weighted_entropy']
        est_bitrates = self.code_bitrates[:,0] * self.loss_weights['weighted_entropy']
        if self.baseline:
            targets_bitrates, est_bitrates = targets_bitrates.sum().double(), est_bitrates.sum().double()
        return torch.abs(est_bitrates - targets_bitrates).sum().double() if self.quant_active else 0.0

    def quantization_loss(self):
        '''
        combine all quantizer modules' quantization error. Used to regularize alpha. 
        don't use if alpha is periodically annealed.
        '''
        return torch.sum(self.quant_losses[:,0]) * self.loss_weights['quant'] if self.quant_active else 0.0
    
    def snr_loss(self, x, y):
        return self.snr(y, x)
    
    def freq_loss(self, x, y):
        return self.freq(y, x) * self.loss_weights['freq']
    
    def loss_fn(self, x ,y, tanh=False):
        snrs, freqs = [],[]
        for i, (xx, yy) in enumerate(zip(x,y)):
            f, s = self.freq_loss(xx, yy), self.snr_loss(xx, yy)
            if i != 0: s = s * self.loss_weights['snr']
            freqs.append(f)
            snrs.append(s)
        ent = self.entropy_loss()
        qua = self.quantization_loss()
        
        snr = torch.stack(snrs).mean()
        freq = torch.stack(freqs).mean()

        return snr+ent+freq, snrs, ent, qua, freqs
    
    def get_overall_entropy_avg(self):
        tar_ent, avgs_ent, avgs_btr = [], [], []
        for i, skip in enumerate(self.skip_encoders):
            sr = self.sr if i > 0 else 16000
            H = self.H if i > 0 else self.H//2
            avgs_ent.append(skip.quant.entropy_avg.avg)
            tar_ent.append(skip.target_entropy)
            avgs_btr.append(utils.entropy_to_bitrate(avgs_ent[-1], sr, H, 64, skip.bottleneck_dims))
        return np.asarray(avgs_ent), np.asarray(avgs_btr)

    def set_network_entropy_target(self, bitrates, fuzzes, sample_rate, frame_size, overlap):
        
        for i, (skip, br, fuzz) in enumerate(zip(self.skip_encoders, bitrates, fuzzes)):
            sr = self.sr if i > 0 else 16000
            H = self.H if i > 0 else self.H//2
            print(skip.bottleneck_dims, utils.bitrate_to_entropy(br,sr,H,overlap,skip.bottleneck_dims))
            skip.target_bitrate = br
            skip.target_entropy = utils.bitrate_to_entropy(br,sr,H,overlap,skip.bottleneck_dims)
            skip.entropy_fuzz = utils.bitrate_to_entropy(fuzz,sr,H,overlap,skip.bottleneck_dims)
            
    def reset_entropy_hists(self):
        for skip in self.skip_encoders:
            skip.quant.entropy_avg.reset()
    
    def on_train_epoch_start(self):
        self.reset_entropy_hists()
        # Check on quant. status
        self.configure_optimizers()
        self.quant_active = True
        for m in self.skip_encoders:
            m.quant_active = True
        
    def training_step(self, batch, batch_idx):
        x, ycb, yhb = batch
        yhhb, yhcb = self(x)
                    
        yhcb_up = self.resampler(yhcb.squeeze(1)).unsqueeze(1)
        
        loss, _, _, _, _ = self.loss_fn([ycb,yhb], [yhcb,yhhb])
        
        return loss
       
    def validation_epoch_start(self):
        self.reset_entropy_hists()
              
    def validation_step(self, batch, batch_idx):
        x, ycb, yhb = batch
        yhhb, yhcb = self(x)
        
        yhcb_up = self.resampler(yhcb.squeeze(1)).unsqueeze(1)
        
        agg, snrs, ent, qua, freqs = self.loss_fn([ycb,yhb], [yhcb,yhhb])
        
        return {'loss':agg, 'snrs':snrs, 'ent':ent, 'qua':qua, 'freqs':freqs}
        
    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([out['loss'] for out in outputs]).mean()
        val_loss_snr = torch.stack([torch.tensor(out['snrs']) for out in outputs]).mean(0)
        val_loss_freq = torch.stack([torch.tensor(out['freqs']) for out in outputs]).mean(0)
        val_loss_ent = torch.stack([torch.tensor(out['ent']) for out in outputs]).mean()
        val_loss_qua = torch.stack([torch.tensor(out['qua']) for out in outputs]).mean()
        entropies, bitrates = self.get_overall_entropy_avg()
        
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss_ent', val_loss_ent, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss_quant', val_loss_qua, on_step=False, on_epoch=True, prog_bar=True)
        self.log('alpha', self.quant_alpha.detach().item(), on_step=False, on_epoch=True, prog_bar=True)
        for i, (e, b) in enumerate(zip(entropies, bitrates)):
            self.log('val_entropy_avg_{}'.format(i), e, on_step=False, on_epoch=True, prog_bar=True)
            self.log('val_bitrate_avg_{}'.format(i), b, on_step=False, on_epoch=True, prog_bar=True)
        for i, (s, f) in enumerate(zip(val_loss_snr, val_loss_freq)):
            self.log('val_loss_snr_{}', s, on_step=False, on_epoch=True, prog_bar=True)
            self.log('val_loss_freq_{}', f, on_step=False, on_epoch=True, prog_bar=True)
            
        self.val_loss_to_console(val_loss, val_loss_snr, val_loss_ent, val_loss_qua, val_loss_freq, entropies, bitrates)
        
        t_bitrate = torch.tensor([skip.target_bitrate for skip in self.skip_encoders]).clone().numpy().astype(np.int)
        e_bitrate = bitrates.astype(np.int)
        if self.baseline:
            t_bitrate, e_bitrate = t_bitrate.sum(), e_bitrate.sum()
        diff = np.abs(t_bitrate-e_bitrate)
        logger.info('Targets Bitrate: {}, Code Bitrate: {}, Diff: {}, Weighted Diff: {} with weight: {}'.format(\
            str(t_bitrate), str(e_bitrate), str(diff), diff * self.loss_weights['weighted_entropy'], self.loss_weights['weighted_entropy']))
    
    def on_test_start(self):
        self.reset_entropy_hists()
     
    def test_step(self, batch, batch_idx):

        self.quant_active = True
        for m in self.skip_encoders:
            m.quant_active = True
            
        x, ycb, yhb = batch
        ycb_32 = self.resampler(ycb.squeeze(0)).unsqueeze(0)

        # batch to time-domain
        yfull_seg, ts_32 = utils.prepare_audio(x, H=self.H, in_chan=self.channel, overlap=32)
        yhb_seg, _ = utils.prepare_audio(yhb, H=self.H, in_chan=self.channel, overlap=32)
        ycb_seg, ts_16 = utils.prepare_audio(ycb, H=self.H//2, in_chan=self.channel, overlap=32)
        ycb_32_seg, _ = utils.prepare_audio(ycb_32, H=self.H, in_chan=self.channel, overlap=32)
        
        # predictions
        yfull_seg = yfull_seg.cpu()
        yhb_seg = yhb_seg.cpu()
        ycb_seg = ycb_seg.cpu()
        ycb_32_seg = ycb_32_seg.cpu()
        
        yhhb, yhcb = self(x.to('cuda'))
        yhcb_32 = self.resampler(yhcb.squeeze(1)).unsqueeze(1)

        # batch to time-domain
        yhhb = utils.overlap_add(yhhb, ts_32, H=self.H, in_chan=self.channel, overlap=32, device='cuda')
        yhcb = utils.overlap_add(yhcb, ts_16, H=self.H//2, in_chan=self.channel, overlap=32, device='cuda')
        yhcb_32 = utils.overlap_add(yhcb_32, ts_32, H=self.H, in_chan=self.channel, overlap=32, device='cuda')
        
        # min calc.
        mn_32 = min(x.shape[-1],ycb_32.shape[-1],yhb.shape[-1],yhhb.shape[-1],yhcb_32.shape[-1])
        
        # get the band-wise snr
        yhhb_filt = torch.from_numpy(utils.fir_high_pass(yhhb[...,:mn_32].detach().cpu().squeeze(), 32000, 8000, 461, np.float32)).reshape(1,-1)
        yhb_filt = torch.from_numpy(utils.fir_high_pass(yhb[...,:mn_32].detach().cpu().squeeze(), 32000, 8000, 461, np.float32)).reshape(1,-1)
        
        if self.hparams.audio_output_path:
            sf.write(os.path.join(self.hparams.audio_output_path,str(batch_idx)+'_input_full.wav'),
                     x.cpu().detach().numpy().squeeze()[:mn_32], samplerate=self.sr, subtype='PCM_16')
            sf.write(os.path.join(self.hparams.audio_output_path,str(batch_idx)+'_pred_full.wav'),
                     yhhb_filt.cpu().detach().numpy().squeeze()[:mn_32]+yhcb_32.cpu().detach().numpy().squeeze()[:mn_32], samplerate=self.sr)
            sf.write(os.path.join(self.hparams.audio_output_path,str(batch_idx)+'_pred_hb.wav'),
                     yhhb_filt.cpu().detach().numpy().squeeze(), samplerate=self.sr)
            sf.write(os.path.join(self.hparams.audio_output_path,str(batch_idx)+'_pred_cb.wav'),
                     yhcb.cpu().detach().numpy().squeeze(), samplerate=self.sr)
            sf.write(os.path.join(self.hparams.audio_output_path,str(batch_idx)+'_input_hb.wav'),
                     yhb_filt.cpu().detach().numpy().squeeze(), samplerate=self.sr)
            sf.write(os.path.join(self.hparams.audio_output_path,str(batch_idx)+'_input_cb.wav'),
                     ycb.cpu().detach().numpy().squeeze(), samplerate=self.sr)

        return {'snr_hb':0.0, 'snr_cb':0.0, 'snr_full':0.0}
    
    def test_epoch_end(self, outputs):

        l_snr_hb = torch.stack([torch.tensor(out['snr_hb']) for out in outputs]).mean(0).item()
        l_snr_cb = torch.stack([torch.tensor(out['snr_cb']) for out in outputs]).mean(0).item()
        l_snr_full = torch.stack([torch.tensor(out['snr_full']) for out in outputs]).mean(0).item()
        entropies, bitrates = self.get_overall_entropy_avg()
        output = dict({'snr_hb':l_snr_hb,'snr_cb':l_snr_cb,'snr_full':l_snr_full,
                       'ent_cb':entropies[0],'br_cb':bitrates[0],
                       'ent_hb':entropies[1],'br_hb':bitrates[1]})
        
        for k, i in output.items():
            logger.info('{} -> {}'.format(k, i))
        
    @rank_zero_only
    def val_loss_to_console(self, val_loss, val_loss_snr, val_loss_ent, val_loss_qua, val_loss_freq, entropy, bitrate):
        logger.info('Epoch {}, Val. Total Loss, {:.4f}, Val. SNR Loss, {}, Val. Entropy Loss (weighted), {:.5f}, Val. Quant Loss, {}, Val. Entropy Avg, {}, Val. Bitrate Avg, {}kbps, Alpha Quant: {}'\
            .format(self.trainer.current_epoch + 1, val_loss, val_loss_snr, val_loss_ent, val_loss_qua/self.loss_weights['quant'], entropy, np.sum(bitrate)/1000, self.quant_alpha))
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
                        self.parameters(), lr=self.lr, weight_decay=self.weight_decay
                    )
        
        return optimizer
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, default_params=None, **kwargs):
        
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        model = cls(**default_params)

        model.load_state_dict(checkpoint['state_dict'], strict=False)
        model.on_load_checkpoint(checkpoint)

        return model
        
        