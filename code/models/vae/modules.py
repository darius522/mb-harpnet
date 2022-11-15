import torch.nn as nn
from loguru import logger

import torch
import torch.nn.functional as F


class Conv1DBuilder(object):

    @staticmethod
    def build(in_channels, out_channels, kernel_size, stride=1, padding=0, use_kaiming_normal=False):
        conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        if use_kaiming_normal:
            conv = nn.utils.weight_norm(conv)
            nn.init.kaiming_normal_(conv.weight)
        return conv
    
class Residual(nn.Module):
    
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens, use_kaiming_normal):
        super(Residual, self).__init__()
        
        relu_1 = nn.ReLU(True)
        conv_1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=num_residual_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        if use_kaiming_normal:
            conv_1 = nn.utils.weight_norm(conv_1)
            nn.init.kaiming_normal_(conv_1.weight)

        relu_2 = nn.ReLU(True)
        conv_2 = nn.Conv1d(
            in_channels=num_residual_hiddens,
            out_channels=num_hiddens,
            kernel_size=1,
            stride=1,
            bias=False
        )
        if use_kaiming_normal:
            conv_2 = nn.utils.weight_norm(conv_2)
            nn.init.kaiming_normal_(conv_2.weight)

        # All parameters same as specified in the paper
        self._block = nn.Sequential(
            relu_1,
            conv_1,
            relu_2,
            conv_2
        )
    
    def forward(self, x):
        return x + self._block(x)
    
class ResidualStack(nn.Module):
    
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, use_kaiming_normal):
        super(ResidualStack, self).__init__()
        
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [Residual(in_channels, num_hiddens, num_residual_hiddens, use_kaiming_normal)] * self._num_residual_layers)
        
    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class ConvolutionalEncoder(nn.Module):
    
    def __init__(self, in_channels=1024, num_hiddens=32, num_residual_layers=2, num_residual_hiddens=2,
        use_kaiming_normal=True, input_features_type=None, features_filters=2, sampling_rate=44100,
        device='cuda', verbose=False):

        super(ConvolutionalEncoder, self).__init__()

        """
        2 preprocessing convolution layers with filter length 3
        and residual connections.
        """

        self._conv_1 = Conv1DBuilder.build(
            in_channels=features_filters,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        self._conv_2 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        """
        1 strided convolution length reduction layer with filter
        length 4 and stride 2 (downsampling the signal by a factor
        of two).
        """
        self._conv_3 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2, # timestep * 2
            use_kaiming_normal=use_kaiming_normal,
            padding=2
        )

        """
        2 convolutional layers with length 3 and
        residual connections.
        """

        self._conv_4 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        self._conv_5 = Conv1DBuilder.build(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=3,
            use_kaiming_normal=use_kaiming_normal,
            padding=1
        )

        """
        4 feedforward ReLu layers with residual connections.
        """

        self._residual_stack = ResidualStack(
            in_channels=num_hiddens,
            num_hiddens=num_hiddens,
            num_residual_layers=num_residual_layers,
            num_residual_hiddens=num_residual_hiddens,
            use_kaiming_normal=use_kaiming_normal
        )
        
        self._input_features_type = input_features_type
        self._features_filters = features_filters
        self._sampling_rate = sampling_rate
        self._device = device
        self._verbose = verbose

    def forward(self, inputs):
        if self._verbose:
            logger.info('inputs size: {}'.format(inputs.size()))

        x_conv_1 = F.relu(self._conv_1(inputs))
        if self._verbose:
            logger.info('x_conv_1 output size: {}'.format(x_conv_1.size()))

        x = F.relu(self._conv_2(x_conv_1)) + x_conv_1
        if self._verbose:
            logger.info('_conv_2 output size: {}'.format(x.size()))
        
        x_conv_3 = F.relu(self._conv_3(x))
        if self._verbose:
            logger.info('_conv_3 output size: {}'.format(x_conv_3.size()))

        x_conv_4 = F.relu(self._conv_4(x_conv_3)) + x_conv_3
        if self._verbose:
            logger.info('_conv_4 output size: {}'.format(x_conv_4.size()))

        x_conv_5 = F.relu(self._conv_5(x_conv_4)) + x_conv_4
        if self._verbose:
            logger.info('x_conv_5 output size: {}'.format(x_conv_5.size()))

        x = self._residual_stack(x_conv_5) + x_conv_5
        if self._verbose:
            logger.info('_residual_stack output size: {}'.format(x.size()))

        return x