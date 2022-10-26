# Base VAE class definition
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from utils import get_mean_param
from utils import Constants
from .distributions import Sparse
from torch.distributions import Laplace, Normal

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

import auraloss
from loguru import logger
from pytorch_lightning.utilities import rank_zero_only

from .objectives import decomp_objective

class PrintLayer(nn.Module):
    def __init__(self, m=''):
        super(PrintLayer, self).__init__()
        self.m = m
    
    def forward(self, x):
        # Do your print / debug stuff here
        #print(self.m,x.shape)
        return x
    
class Enc(nn.Module):
    """ https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN/blob/master/pytorch_CelebA_DCGAN.py """
    def __init__(self, in_chan, base, latent_dim):
        super(Enc, self).__init__()
        self.enc = nn.Sequential(
            # input size is in_size x 512
            nn.Conv1d(in_chan, base, 4, 4, 1, bias=False),
            nn.ReLU(inplace=True),
            PrintLayer(m='1: '),
            # state size: ndf x 256
            nn.Conv1d(base, base * 2, 4, 4, 1, bias=False),
            nn.BatchNorm1d(base * 2),
            nn.ReLU(inplace=True),
            PrintLayer(m='2: '),
            # state size: (ndf * 2) x 64
            nn.Conv1d(base * 2, base * 4, 4, 8, 1, bias=False),
            nn.BatchNorm1d(base * 4),
            nn.ReLU(inplace=True),
            PrintLayer(m='3: '),
            # state size: (ndf * 4) x 8
        )
        self.c1 = nn.Conv1d(base * 4, latent_dim, 4, 5)
        self.c2 = nn.Conv1d(base * 4, latent_dim, 4, 5)
        # c1, c2 size: latent_dim x 1

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        e = self.enc(x)
        mu = self.c1(e)
        scale = F.softplus(self.c2(e)).view_as(mu) + Constants.eta
        return mu.squeeze(-1), scale.squeeze(-1)

class Dec(nn.Module):
    """ https://github.com/znxlwm/pytorch-MNIST-CelebA-GAN-DCGAN/blob/master/pytorch_CelebA_DCGAN.py
        https://github.com/seangal/dcgan_vae_pytorch/blob/master/main.py
        https://github.com/last-one/DCGAN-Pytorch/blob/master/network.py
    """
    def __init__(self, in_chan, base, latent_dim):
        super(Dec, self).__init__()
        self.dec = nn.Sequential(
            # input size is z_size
            nn.ConvTranspose1d(latent_dim, base * 4, 4, 8, 1, bias=False),
            nn.BatchNorm1d(base * 4),
            nn.ReLU(inplace=True),
            PrintLayer(m='1: '),
            # state size: (ngf * 8) x 4 x 4
            nn.ConvTranspose1d(base * 4, base * 2, 4, 8, 1, bias=False),
            nn.BatchNorm1d(base * 2),
            nn.ReLU(inplace=True),
            PrintLayer(m='2: '),
            # state size: (ngf * 4) x 8 x 8
            nn.ConvTranspose1d(base * 2, base, 4, 8, 1, bias=False),
            nn.BatchNorm1d(base),
            nn.ReLU(inplace=True),
            PrintLayer(m='3: '),
            # state size: (ngf * 2) x 16 x 16
            nn.ConvTranspose1d(base, in_chan, 4, 14, 1, bias=False),
            PrintLayer(m='4: '),
            # state size: out_size x 64 x 64
        )
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, z):
        #z = z.view(-1, z.size(-1), 1, 1)
        d = self.dec(z)
        #d = d.view(*pre_shape, *torch.Size([1, 32, 32]))
        return d, torch.tensor(0.1).to(z.device)  # or 0.05

class VAE(LightningModule):
    def __init__(self, in_chan=2, base=32, latent_dim=10, prior_variance='iso',
                 lr = 0.001, patience = 20, lr_decay_patience = 5, lr_decay_gamma = 0.3, weight_decay = 0.00001, **kwargs):
        super(VAE, self).__init__()
        
        self.lr = lr
        self.patience = patience
        self.lr_decay_patience = lr_decay_patience
        self.lr_decay_gamma = lr_decay_gamma
        self.weight_decay = weight_decay
        
        self.pz = Normal #prior_dist
        self.px_z = Normal #likelihood_dist
        self.qz_x = Normal #posterior_dist
        
        self.enc = Enc(in_chan=2, base=32, latent_dim=10)
        self.dec = Dec(in_chan=2, base=32, latent_dim=10)
        
        self.loss_fn = decomp_objective
        self.snr = auraloss.time.SNRLoss()
        
        self.latent_dim = latent_dim
        self.prior_variance = prior_variance
        self.beta = 1.0
        self.alpha = 0.0
        self.gamma = 0.8
        self.df = 2.0
        self.K = 1

        self._pz_mu, self._pz_logvar = self.init_pz(latent_dim=self.latent_dim, prior_variance=prior_variance)
        self.prior_variance_scale = 1.0
        self.gamma = nn.Parameter(torch.tensor(self.gamma), requires_grad=False)
        self.df = nn.Parameter(torch.tensor(self.df), requires_grad=False)
        print('p(z):')
        print(self.pz)
        print(self.pz_params)
        print('q(z|x):')
        print(self.qz_x)

        if self.px_z == dist.RelaxedBernoulli:
            self.px_z.log_prob = lambda self, value: \
                -F.binary_cross_entropy_with_logits(
                    self.probs if value.dim() <= self.probs.dim() else self.probs.expand_as(value),
                    value.expand(self.batch_shape) if value.dim() <= self.probs.dim() else value,
                    reduction='none'
                )

    @property
    def device(self):
        return self._pz_mu.device

    def generate(self, N, K):
        self.eval()
        with torch.no_grad():
            mean_pz = get_mean_param(self.pz_params)
            mean = get_mean_param(self.dec(mean_pz))
            pz = self.pz(*self.pz_params)
            if self.pz == torch.distributions.studentT.StudentT:
                pz._chi2 = torch.distributions.Chi2(pz.df)  # fix from rsample
            px_z_params = self.dec(pz.sample(torch.Size([N])))
            means = get_mean_param(px_z_params)
            samples = self.px_z(*px_z_params).sample(torch.Size([K]))

        return mean, \
            means.view(-1, *means.size()[2:]), \
            samples.view(-1, *samples.size()[3:])

    def reconstruct(self, data):
        self.eval()
        with torch.no_grad():
            qz_x = self.qz_x(*self.enc(data))
            px_z_params = self.dec(qz_x.rsample().unsqueeze(-1))

        return get_mean_param(px_z_params)

    def forward(self, x, K=1, no_dec=False):
        qz_x = self.qz_x(*self.enc(x))
        zs = qz_x.rsample(torch.Size([K]))
        if no_dec:
            return qz_x, zs
        px_z = self.px_z(*self.dec(zs.permute(1,2,0)))
        return qz_x, px_z, zs

    @property
    def pz_params(self):
        mu = self._pz_mu.mul(1)
        scale = torch.sqrt(self.prior_variance_scale * self._pz_logvar.size(-1) * F.softmax(self._pz_logvar, dim=1))
        return mu, scale

    def init_pz(self, latent_dim, prior_variance, learn_prior_variance=False):
        # means
        pz_mu = nn.Parameter(torch.zeros(1, latent_dim), requires_grad=False)

        # variances
        if prior_variance == 'iso':
            logvar = torch.zeros(1, latent_dim)
        elif prior_variance == 'pca':
            singular_values = self.dataset.load_pca(latent_dim).log()
            logvar = singular_values.expand(1, latent_dim)
        pz_logvar = nn.Parameter(logvar, requires_grad=learn_prior_variance)

        return pz_mu, pz_logvar

    def posterior_plot(self, zs_mean, zs_std, runPath, epoch):
        pass
    
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        p, q, zs = self(x, K=1)
        loss = -self.loss_fn(self, x, K=self.K, beta=self.beta, alpha=self.alpha, regs=None)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = -self.loss_fn(self, x, K=self.K, beta=self.beta, alpha=self.alpha, regs=None)
        self.log("val_loss", loss, on_epoch=True)
        
        yhat = self.reconstruct(x)
        print('haha',yhat.shape)
        logger.info('SNR: {}'.format(self.snr(yhat, x)))
        return loss
        
    def validation_epoch_end(self, outputs):
        val_loss = torch.stack(outputs).mean()
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_loss_to_console(val_loss)
        
    @rank_zero_only
    def val_loss_to_console(self, val_loss):
        logger.info('Epoch {}, Validation Loss, {:.6f} '.format(self.trainer.current_epoch + 1, val_loss))
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
                        self.parameters(), lr=self.lr, weight_decay=self.weight_decay
                    )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, factor=self.lr_decay_gamma, patience=self.lr_decay_patience, cooldown=10
                    )
        
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
    
    def configure_callbacks(self):
        checkpoint = ModelCheckpoint(monitor="val_loss")
        return [checkpoint]
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, default_params=None, **kwargs):
        import yaml
        from argparse import Namespace
        
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        default_hyperparams = yaml.safe_load(open(default_params))

        hparams = Namespace(**default_hyperparams)
        model = cls(hparams)

        model.load_state_dict(checkpoint['state_dict'])
        model.on_load_checkpoint(checkpoint)

        return model
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)