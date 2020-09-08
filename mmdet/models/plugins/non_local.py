import torch
import torch.nn as nn
from mmcv.cnn import constant_init, normal_init

from ..utils import ConvModule
from mmdet.ops import ContextBlock

from torch.nn.parameter import Parameter

class NonLocal2D(nn.Module):
    """Non-local module.

    See https://arxiv.org/abs/1711.07971 for details.

    Args:
        in_channels (int): Channels of the input feature map.
        reduction (int): Channel reduction ratio.
        use_scale (bool): Whether to scale pairwise_weight by 1/inter_channels.
        conv_cfg (dict): The config dict for convolution layers.
            (only applicable to conv_out)
        norm_cfg (dict): The config dict for normalization layers.
            (only applicable to conv_out)
        mode (str): Options are `embedded_gaussian` and `dot_product`.
    """

    def __init__(self,
                 in_channels,
                 reduction=2,
                 use_scale=True,
                 conv_cfg=None,
                 norm_cfg=None,
                 mode='embedded_gaussian',
                 whiten_type=None,
                 temp=1.0,
                 downsample=False,
                 fixbug=False,
                 learn_t=False,
                 gcb=None):
        super(NonLocal2D, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = in_channels // reduction
        self.mode = mode
        assert mode in ['embedded_gaussian', 'dot_product', 'gaussian']
        if mode == 'gaussian':
            self.with_embedded = False
        else:
            self.with_embedded = True
        self.whiten_type = whiten_type
        assert whiten_type in [None, 'channel', 'bn-like']  # TODO: support more
        self.learn_t = learn_t
        if self.learn_t:
            self.temp = Parameter(torch.Tensor(1))
            self.temp.data.fill_(temp)
        else:
            self.temp = temp
        if downsample:
            self.downsample = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        else:
            self.downsample = None
        self.fixbug=fixbug

        assert gcb is None or isinstance(gcb, dict)
        self.gcb = gcb
        if gcb is not None:
            self.gc_block = ContextBlock(inplanes=in_channels, **gcb)
        else:
            self.gc_block = None

        # g, theta, phi are actually `nn.Conv2d`. Here we use ConvModule for
        # potential usage.
        self.g = ConvModule(
            self.in_channels,
            self.inter_channels,
            kernel_size=1,
            activation=None)
        if self.with_embedded:
            self.theta = ConvModule(
                self.in_channels,
                self.inter_channels,
                kernel_size=1,
                activation=None)
            self.phi = ConvModule(
                self.in_channels,
                self.inter_channels,
                kernel_size=1,
                activation=None)
        self.conv_out = ConvModule(
            self.inter_channels,
            self.in_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            activation=None)

        self.init_weights()

    def init_weights(self, std=0.01, zeros_init=True):
        transform_list = [self.g]
        if self.with_embedded:
            transform_list.extend([self.theta, self.phi])
        for m in transform_list:
            normal_init(m.conv, std=std)
        if zeros_init:
            constant_init(self.conv_out.conv, 0)
        else:
            normal_init(self.conv_out.conv, std=std)

    def embedded_gaussian(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            # theta_x.shape[-1] is `self.inter_channels`
            if self.fixbug:
                pairwise_weight /= theta_x.shape[-1]**0.5
            else:
                pairwise_weight /= theta_x.shape[-1]**-0.5
        if self.learn_t:
            pairwise_weight = pairwise_weight * nn.functional.softplus(self.temp) # stable training
        else:
            pairwise_weight = pairwise_weight / self.temp
        pairwise_weight = pairwise_weight.softmax(dim=-1)
        return pairwise_weight

    def gaussian(self, theta_x, phi_x):
        return self.embedded_gaussian(theta_x, phi_x)

    def dot_product(self, theta_x, phi_x):
        # pairwise_weight: [N, HxW, HxW]
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight /= pairwise_weight.shape[-1]
        return pairwise_weight

    def forward(self, x):
        n, _, h, w = x.shape
        if self.downsample:
            down_x = self.downsample(x)
        else:
            down_x = x

        # g_x: [N, H'xW', C], VALUE?
        g_x = self.g(down_x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        # theta_x: [N, HxW, C], QUERY?
        if self.with_embedded:
            theta_x = self.theta(x).view(n, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
        else:
            theta_x = x.view(n, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)

        # phi_x: [N, C, H'xW'], KEY?
        if self.with_embedded:
            phi_x = self.phi(down_x).view(n, self.inter_channels, -1)
        else:
            phi_x = x.view(n, self.in_channels, -1)

        # whiten
        if self.whiten_type == "channel":
            theta_x_mean = theta_x.mean(2).unsqueeze(2)
            phi_x_mean = phi_x.mean(2).unsqueeze(2)
            theta_x -= theta_x_mean
            phi_x -= phi_x_mean
        elif self.whiten_type == 'bn-like':
            theta_x_mean = theta_x.mean(2).mean(0).unsqueeze(0).unsqueeze(2)
            phi_x_mean = phi_x.mean(2).mean(0).unsqueeze(0).unsqueeze(2)
            theta_x -= theta_x_mean
            phi_x -= phi_x_mean

        pairwise_func = getattr(self, self.mode)
        # pairwise_weight: [N, HxW, H'xW']
        pairwise_weight = pairwise_func(theta_x, phi_x)

        # y: [N, HxW, C]
        y = torch.matmul(pairwise_weight, g_x)
        # y: [N, C, H, W]
        y = y.permute(0, 2, 1).reshape(n, self.inter_channels, h, w)


        # gc block
        if self.gcb:
            output = self.gc_block(x) + self.conv_out(y)
        else:
            output = x + self.conv_out(y)

        return output
