import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
import math
import inplace_abn


class _NonLocalNd_bn(nn.Module):
    def __init__(self, dim, inplanes, ratio, downsample, use_gn, lr_mult, use_out, out_bn, whiten_type, temperature, with_gc, with_2fc, double_conv):
        assert dim in [1, 2, 3], "dim {} is not supported yet".format(dim)
        #assert whiten_type in ['channel', 'spatial']
        if dim == 3:
            conv_nd = nn.Conv3d
            if downsample:
                max_pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
            else:
                max_pool = None
            bn_nd = nn.BatchNorm3d
        elif dim == 2:
            conv_nd = nn.Conv2d
            if downsample:
                max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
            else:
                max_pool = None
            bn_nd = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            if downsample:
                max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
            else:
                max_pool = None
            bn_nd = nn.BatchNorm1d
        planes = int(inplanes * ratio)
        self.planes = planes
        super(_NonLocalNd_bn, self).__init__()
        if double_conv:
            self.conv_query = nn.Sequential(
                conv_nd(inplanes, planes, kernel_size=1),
                inplace_abn.InPlaceABNSync(planes, activation="identity"),
                nn.ReLU(inplace=False),
                conv_nd(planes, planes, kernel_size=1))
            self.conv_key = nn.Sequential(
                conv_nd(inplanes, planes, kernel_size=1),
                inplace_abn.InPlaceABNSync(planes, activation="identity"),
                nn.ReLU(inplace=False),
                conv_nd(planes, planes, kernel_size=1))
        else:
            self.conv_query = conv_nd(inplanes, planes, kernel_size=1)
            self.conv_key = conv_nd(inplanes, planes, kernel_size=1)
        if use_out:
            self.conv_value = conv_nd(inplanes, planes, kernel_size=1)
            self.conv_out = conv_nd(planes, inplanes, kernel_size=1, bias=False)
        else:
            self.conv_value = conv_nd(inplanes, inplanes, kernel_size=1, bias=False)
            self.conv_out = None
        if out_bn:
            self.out_bn = nn.BatchNorm2d(inplanes)
        else:
            self.out_bn = None
        self.with_2fc = with_2fc
        if with_gc:
            self.conv_mask = conv_nd(inplanes, 1, kernel_size=1)
            if self.with_2fc:
                self.channel_mul_conv = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=1),
                    nn.LayerNorm([planes, 1, 1]),
                    nn.ReLU(inplace=True),  # yapf: disable
                    nn.Conv2d(planes, inplanes, kernel_size=1))
        if 'bn_affine' in whiten_type:
            self.key_bn_affine = nn.BatchNorm1d(planes)
            self.query_bn_affine = nn.BatchNorm1d(planes)
        if 'bn' in whiten_type:
            self.key_bn = nn.BatchNorm1d(planes, affine=False)
            self.query_bn = nn.BatchNorm1d(planes, affine=False)
        self.softmax = nn.Softmax(dim=2)
        self.downsample = max_pool
        self.gamma = nn.Parameter(torch.zeros(1))
        self.scale = math.sqrt(planes)
        self.whiten_type = whiten_type
        self.temperature = temperature
        self.with_gc = with_gc
        
        self.reset_parameters()
        self.reset_lr_mult(lr_mult)

    def reset_parameters(self):

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    init.zeros_(m.bias)
                m.inited = True
        #init.constant_(self.norm.weight, 0)
        #init.constant_(self.norm.bias, 0)
        #self.norm.inited = True

    def reset_lr_mult(self, lr_mult):
        if lr_mult is not None:
            for m in self.modules():
                m.lr_mult = lr_mult
        else:
            print('not change lr_mult')

    def forward(self, x):
        # [N, C, T, H, W]
        residual = x
        # [N, C, T, H', W']
        if self.downsample is not None:
            input_x = self.downsample(x)
        else:
            input_x = x

        # [N, C', T, H, W]
        query = self.conv_query(x)
        # [N, C', T, H', W']
        key = self.conv_key(input_x)
        value = self.conv_value(input_x)

        # [N, C', H x W]
        query = query.view(query.size(0), query.size(1), -1)
        # [N, C', H' x W']
        key = key.view(key.size(0), key.size(1), -1)
        value = value.view(value.size(0), value.size(1), -1)
        
        if 'channel' in self.whiten_type :
            key_mean = key.mean(2).unsqueeze(2)
            query_mean = query.mean(2).unsqueeze(2)
            key -= key_mean
            query -= query_mean
        if 'spatial' in self.whiten_type :
            key_mean = key.mean(1).unsqueeze(1)
            query_mean = query.mean(1).unsqueeze(1)
            key -= key_mean
            query -= query_mean
        if 'bn_affine' in self.whiten_type :
            key = self.key_bn_affine(key)
            query = self.query_bn_affine(query)
        if 'bn' in self.whiten_type:
            key = self.key_bn(key)
            query = self.query_bn(query)

        # [N, T x H x W, T x H' x W']
        sim_map = torch.bmm(query.transpose(1, 2), key)
        sim_map = sim_map/self.scale
        sim_map = sim_map/self.temperature
        sim_map = self.softmax(sim_map)

        # [N, T x H x W, C']
        out_sim = torch.bmm(sim_map, value.transpose(1, 2))
        # [N, C', T x H x W]
        out_sim = out_sim.transpose(1, 2)
        # [N, C', T,  H, W]
        out_sim = out_sim.view(out_sim.size(0), out_sim.size(1), *x.size()[2:])
        # if self.norm is not None:
        #     out = self.norm(out)
        out_sim = self.gamma * out_sim
        
        #out = residual + out_sim
        
        if self.with_gc:
            if self.with_2fc:
                # [N, 1, H', W']
                mask = self.conv_mask(input_x)
                # [N, 1, H'x W']
                mask = mask.view(mask.size(0), mask.size(1), -1)
                mask = self.softmax(mask)
                # [N, 1, H'x W', 1]
                mask = mask.unsqueeze(-1)
                # [N, C, H'x W']
                input_x = input_x.view(input_x.size(0), input_x.size(1), -1)
                # [N, 1, C, H'x W']
                input_x = input_x.unsqueeze(1)
                # [N, 1, C, 1]
                out_gc = torch.matmul(input_x, mask)
                # [N, C, 1, 1]
                out_gc = out_gc.view(out_gc.size(0), out_gc.size(2), 1, 1)
                out_gc = self.channel_mul_conv(out_gc)
                out_sim = out_sim+out_gc
            else:
                # [N, 1, H', W']
                mask = self.conv_mask(input_x)
                # [N, 1, H'x W']
                mask = mask.view(mask.size(0), mask.size(1), -1)
                mask = self.softmax(mask)
                # [N, C, 1, 1]
                out_gc = torch.bmm(value, mask.permute(0,2,1)).unsqueeze(-1)
                out_sim = out_sim+out_gc
            
        # [N, C, T,  H, W]
        if self.conv_out is not None:
            out_sim = self.conv_out(out_sim)
        if self.out_bn:
            out_sim = self.out_bn(out_sim)
            
        out = out_sim + residual

        return out


class NonLocal2dGc(_NonLocalNd_bn):
    def __init__(self, inplanes, ratio, downsample=True, use_gn=False, lr_mult=None, use_out=False, out_bn=False, whiten_type=['channel'], temp=1.0, with_gc=False, with_2fc=False, double_conv=False):
        super(NonLocal2dGc, self).__init__(dim=2, inplanes=inplanes, ratio=ratio, downsample=downsample, use_gn=use_gn, lr_mult=lr_mult, use_out=use_out, out_bn=out_bn, whiten_type=whiten_type, temperature=temp, with_gc=with_gc, with_2fc=with_2fc, double_conv=double_conv)


class NonLocal3dGc(_NonLocalNd_bn):
    def __init__(self, inplanes, ratio, downsample, use_gn, lr_mult, use_out):
        super(NonLocal3dGc, self).__init__(dim=3, inplanes=inplanes, ratio=ratio, downsample=downsample, use_gn=use_gn, lr_mult=lr_mult, use_out=use_out)
