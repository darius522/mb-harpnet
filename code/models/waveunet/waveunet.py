import torch
import torch.nn as nn

from .crop import centre_crop
from .resample import Resample1d
from .conv import ConvLayer

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

import auraloss
from loguru import logger
from pytorch_lightning.utilities import rank_zero_only

import numpy as np
import soundfile as sf
import os

class UpsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, depth, conv_type, res):
        super(UpsamplingBlock, self).__init__()
        assert(stride > 1)

        # CONV 1 for UPSAMPLING
        if res == "fixed":
            self.upconv = Resample1d(n_inputs, 15, stride, transpose=True)
        else:
            self.upconv = ConvLayer(n_inputs, n_inputs, kernel_size, stride, conv_type, transpose=True)

        self.pre_shortcut_convs = nn.ModuleList([ConvLayer(n_inputs, n_outputs, kernel_size, 1, conv_type)] +
                                                [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in range(depth - 1)])

        # CONVS to combine high- with low-level information (from shortcut)
        self.post_shortcut_convs = nn.ModuleList([ConvLayer(n_outputs + n_shortcut, n_outputs, kernel_size, 1, conv_type)] +
                                                 [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in range(depth - 1)])

    def forward(self, x, shortcut):
        # UPSAMPLE HIGH-LEVEL FEATURES
        upsampled = self.upconv(x)

        for conv in self.pre_shortcut_convs:
            upsampled = conv(upsampled)

        # Prepare shortcut connection
        combined = centre_crop(shortcut, upsampled)

        # Combine high- and low-level features
        for conv in self.post_shortcut_convs:
            combined = conv(torch.cat([combined, centre_crop(upsampled, combined)], dim=1))
        return combined

    def get_output_size(self, input_size):
        curr_size = self.upconv.get_output_size(input_size)

        # Upsampling convs
        for conv in self.pre_shortcut_convs:
            curr_size = conv.get_output_size(curr_size)

        # Combine convolutions
        for conv in self.post_shortcut_convs:
            curr_size = conv.get_output_size(curr_size)

        return curr_size

class DownsamplingBlock(nn.Module):
    def __init__(self, n_inputs, n_shortcut, n_outputs, kernel_size, stride, depth, conv_type, res):
        super(DownsamplingBlock, self).__init__()
        assert(stride > 1)

        self.kernel_size = kernel_size
        self.stride = stride

        # CONV 1
        self.pre_shortcut_convs = nn.ModuleList([ConvLayer(n_inputs, n_shortcut, kernel_size, 1, conv_type)] +
                                                [ConvLayer(n_shortcut, n_shortcut, kernel_size, 1, conv_type) for _ in range(depth - 1)])

        self.post_shortcut_convs = nn.ModuleList([ConvLayer(n_shortcut, n_outputs, kernel_size, 1, conv_type)] +
                                                 [ConvLayer(n_outputs, n_outputs, kernel_size, 1, conv_type) for _ in
                                                  range(depth - 1)])

        # CONV 2 with decimation
        if res == "fixed":
            self.downconv = Resample1d(n_outputs, 15, stride) # Resampling with fixed-size sinc lowpass filter
        else:
            self.downconv = ConvLayer(n_outputs, n_outputs, kernel_size, stride, conv_type)

    def forward(self, x):
        # PREPARING SHORTCUT FEATURES
        shortcut = x
        for conv in self.pre_shortcut_convs:
            shortcut = conv(shortcut)

        # PREPARING FOR DOWNSAMPLING
        out = shortcut
        for conv in self.post_shortcut_convs:
            out = conv(out)

        # DOWNSAMPLING
        out = self.downconv(out)

        return out, shortcut

    def get_input_size(self, output_size):
        curr_size = self.downconv.get_input_size(output_size)

        for conv in reversed(self.post_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)

        for conv in reversed(self.pre_shortcut_convs):
            curr_size = conv.get_input_size(curr_size)
        return curr_size

class Waveunet(LightningModule):
    def __init__(self, 
                 num_inputs=1, 
                 num_channels=[32,64], 
                 num_outputs=1, 
                 instruments=["bass", "drums", "other", "vocals"], 
                 kernel_size=15, 
                 target_output_size=16384, 
                 conv_type="gn", 
                 res="fixed", separate=False, depth=2, strides=4):
        super(Waveunet, self).__init__()

        self.num_levels = len(num_channels)
        self.strides = strides
        self.kernel_size = kernel_size
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.depth = depth
        self.instruments = instruments
        self.separate = separate

        # Only odd filter kernels allowed
        assert(kernel_size % 2 == 1)

        self.waveunets = nn.ModuleDict()

        model_list = instruments if separate else ["ALL"]
        # Create a model for each source if we separate sources separately, otherwise only one (model_list=["ALL"])
        for instrument in model_list:
            module = nn.Module()

            module.downsampling_blocks = nn.ModuleList()
            module.upsampling_blocks = nn.ModuleList()

            for i in range(self.num_levels - 1):
                in_ch = num_inputs if i == 0 else num_channels[i]

                module.downsampling_blocks.append(
                    DownsamplingBlock(in_ch, num_channels[i], num_channels[i+1], kernel_size, strides, depth, conv_type, res))

            for i in range(0, self.num_levels - 1):
                module.upsampling_blocks.append(
                    UpsamplingBlock(num_channels[-1-i], num_channels[-2-i], num_channels[-2-i], kernel_size, strides, depth, conv_type, res))

            module.bottlenecks = nn.ModuleList(
                [ConvLayer(num_channels[-1], num_channels[-1], kernel_size, 1, conv_type) for _ in range(depth)])

            # Output conv
            outputs = num_outputs if separate else num_outputs * len(instruments)
            module.output_conv = nn.Conv1d(num_channels[0], outputs, 1)

            self.waveunets[instrument] = module

        self.set_output_size(target_output_size)

    def set_output_size(self, target_output_size):
        self.target_output_size = target_output_size

        self.input_size, self.output_size = self.check_padding(target_output_size)
        print("Using valid convolutions with " + str(self.input_size) + " inputs and " + str(self.output_size) + " outputs")

        assert((self.input_size - self.output_size) % 2 == 0)
        self.shapes = {"output_start_frame" : (self.input_size - self.output_size) // 2,
                       "output_end_frame" : (self.input_size - self.output_size) // 2 + self.output_size,
                       "output_frames" : self.output_size,
                       "input_frames" : self.input_size}

    def check_padding(self, target_output_size):
        # Ensure number of outputs covers a whole number of cycles so each output in the cycle is weighted equally during training
        bottleneck = 1

        while True:
            out = self.check_padding_for_bottleneck(bottleneck, target_output_size)
            if out is not False:
                return out
            bottleneck += 1

    def check_padding_for_bottleneck(self, bottleneck, target_output_size):
        module = self.waveunets[[k for k in self.waveunets.keys()][0]]
        try:
            curr_size = bottleneck
            for idx, block in enumerate(module.upsampling_blocks):
                curr_size = block.get_output_size(curr_size)
            output_size = curr_size

            # Bottleneck-Conv
            curr_size = bottleneck
            for block in reversed(module.bottlenecks):
                curr_size = block.get_input_size(curr_size)
            for idx, block in enumerate(reversed(module.downsampling_blocks)):
                curr_size = block.get_input_size(curr_size)

            assert(output_size >= target_output_size)
            return curr_size, output_size
        except AssertionError as e:
            return False

    def forward_module(self, x, module):
        '''
        A forward pass through a single Wave-U-Net (multiple Wave-U-Nets might be used, one for each source)
        :param x: Input mix
        :param module: Network module to be used for prediction
        :return: Source estimates
        '''
        shortcuts = []
        out = x
        print('in: ', x.shape)
        # DOWNSAMPLING BLOCKS
        for block in module.downsampling_blocks:
            out, short = block(out)
            print("down: ", out.shape, short.shape)
            shortcuts.append(short)

        # BOTTLENECK CONVOLUTION
        for conv in module.bottlenecks:
            out = conv(out)
            print('bl: ', out.shape)

        # UPSAMPLING BLOCKS
        for idx, block in enumerate(module.upsampling_blocks):
            out = block(out, shortcuts[-1 - idx])
            print('up: ', out.shape)

        # OUTPUT CONV
        out = module.output_conv(out)
        print('out conv: ', out.shape)
        if not self.training:  # At test time clip predictions to valid amplitude range
            out = out.clamp(min=-1.0, max=1.0)
        return out

    def forward(self, x, inst=None):
        curr_input_size = x.shape[-1]
        assert(curr_input_size == self.input_size) # User promises to feed the proper input himself, to get the pre-calculated (NOT the originally desired) output size

        if self.separate:
            return {inst : self.forward_module(x, self.waveunets[inst])}
        else:
            assert(len(self.waveunets) == 1)
            out = self.forward_module(x, self.waveunets["ALL"])

            out_dict = {}
            for idx, inst in enumerate(self.instruments):
                out_dict[inst] = out[:, idx * self.num_outputs:(idx + 1) * self.num_outputs]
            return out_dict
        
    
    def snr_loss(self, x, y):
        return self.snr(y, x)
    
    def loss_fn(self, x ,y):
        snr = self.snr_loss(y, x)
        freq = self.freq(y, x) * 20.0
        ent = self.entropy_loss()
        qua = self.quantization_loss()
        
        return snr+ent+qua+freq, snr, ent, qua
            
    def training_step(self, batch, batch_idx):
        x, y = batch
        yhat = self(x)
        mn = min(y.shape[-1],yhat.shape[-1])
        loss, _, _, _ = self.loss_fn(yhat[...,:mn], y[...,:mn])
                
        return loss
    
    def training_epoch_end(self, training_step_outputs):   
        self.reset_entropy_hists()
        # Check on quant. status
        if self.current_epoch == self.quant_epoch:
            self.configure_optimizers()
            self.configure_callbacks(monitor='val_loss')
            logger.info('Quantization Became Active!')
            self.quant_active = True
            for m in self.skip_encoders:
                m.quant_active = True
                    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        yhat = self.forward(x)
        mn = min(y.shape[-1],yhat.shape[-1])
        loss, snr, ent, qua = self.loss_fn(yhat[...,:mn], y[...,:mn])
        
        return {'loss':loss, 'snr':snr, 'ent':ent, 'qua':qua}
        
    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([out['loss'] for out in outputs]).mean()
        val_loss_snr = torch.stack([out['snr'] for out in outputs]).mean()
        val_loss_ent = torch.stack([torch.tensor(out['ent']) for out in outputs]).mean()
        val_loss_qua = torch.stack([torch.tensor(out['qua']) for out in outputs]).mean()
        entropy, bitrate = self.get_overall_entropy_avg()
        
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss_snr', val_loss_snr, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss_ent', val_loss_ent, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_entropy_avg', entropy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_bitrate_avg', bitrate, on_step=False, on_epoch=True, prog_bar=True)
        self.val_loss_to_console(val_loss, val_loss_snr, val_loss_ent, val_loss_qua, entropy, bitrate)
        
        self.entropy_control_update()
        self.reset_entropy_hists()
    
    def on_test_start(self):
        pass
     
    def test_step(self, batch, batch_idx):

        self.quant_active = True
        for m in self.skip_encoders:
            m.quant_active = True
            
        x, y = batch
        x_seg, ts = utils.prepare_audio(x, H=self.H, in_chan=self.channel, overlap=0.5)
        y_seg, _ = utils.prepare_audio(y, H=self.H, in_chan=self.channel, overlap=0.5)
        
        x_seg = x_seg.to(x.get_device())
        y_seg = y_seg.to(y.get_device())
        
        y_hat = self.forward(x_seg)
        
        y = utils.overlap_add(y_seg, ts, H=self.H, in_chan=self.channel, overlap=0.5)
        y_hat = utils.overlap_add(y_hat, ts, H=self.H, in_chan=self.channel, overlap=0.5)
        y_hat /= self.scaler
        y /= self.scaler
        
        snr = -self.snr_loss(y_hat, y)
        
        if self.hparams.audio_output_path and batch_idx < 4:
            sf.write(os.path.join(self.hparams.audio_output_path,str(batch_idx)+'_y.wav'),
                     y.cpu().detach().numpy().squeeze(), samplerate=self.sr)
            sf.write(os.path.join(self.hparams.audio_output_path,str(batch_idx)+'_yh.wav'),
                     y_hat.cpu().detach().numpy().squeeze(), samplerate=self.sr)
            
        entropy, bitrate = self.get_overall_entropy_avg()
        print(bitrate, entropy, self.sr, self.H, self.H//2, self.bottleneck_dims)
        
        self.reset_entropy_hists()
        
        output = dict({
                'test_loss': snr,
                'test_entropy': torch.tensor(entropy),
                'test_bitrate': torch.tensor(bitrate),
                })
    
        return output
    

    def test_epoch_end(self, test_step_outputs):
        output = dict()
        for r in test_step_outputs:
            for k, i in r.items():
                output.setdefault(k,[]).append(i)
            
        for k, i in output.items():
            logger.info('{} -> {}'.format(k, torch.tensor(i).mean()))
        
    @rank_zero_only
    def val_loss_to_console(self, val_loss, val_loss_snr, val_loss_ent, val_loss_qua, entropy, bitrate):
        logger.info('Epoch {}, Val. Total Loss, {:.4f}, Val. SNR Loss, {:.4f}, Val. Entropy Loss (weighted), {:.5f},\
            Val. Quant Loss (weighted), {:.5f}, Val. Entropy Avg, {:.5f}, Val. Bitrate Avg, {}kbps'\
            .format(self.trainer.current_epoch + 1, val_loss, val_loss_snr, val_loss_ent, val_loss_qua, entropy, int(bitrate/1000)))
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
                        self.parameters(), lr=self.lr, weight_decay=self.weight_decay
                    )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, factor=self.lr_decay_gamma, patience=self.lr_decay_patience, cooldown=10, verbose=True
                    )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
    
    def configure_callbacks(self, monitor=None):
        checkpoint = ModelCheckpoint(monitor=monitor)
        return [checkpoint]
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, default_params=None, **kwargs):
        import yaml
        from argparse import Namespace
        
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

        model = cls(**default_params)

        model.load_state_dict(checkpoint['state_dict'])
        model.on_load_checkpoint(checkpoint)

        return model