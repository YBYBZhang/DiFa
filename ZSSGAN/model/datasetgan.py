# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from numpy import number
from collections import OrderedDict
import torch
import torch.nn as nn


class pixel_classifier(nn.Module):
    def __init__(self, model_path, numpy_class, dim, args=None):
        super(pixel_classifier, self).__init__()
        if numpy_class < 32:
            self.layers = nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=32),
                nn.Linear(32, numpy_class),
                # nn.Sigmoid()
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, numpy_class),
                # nn.Sigmoid()
            )
        mode = 'bilinear'
        self.out_res = 256
        self.upsamplers = [nn.Upsample(scale_factor=self.out_res / 4, mode=mode),
                  nn.Upsample(scale_factor=self.out_res / 4, mode=mode),
                  nn.Upsample(scale_factor=self.out_res / 8, mode=mode),
                  nn.Upsample(scale_factor=self.out_res / 8, mode=mode),
                  nn.Upsample(scale_factor=self.out_res / 16, mode=mode),
                  nn.Upsample(scale_factor=self.out_res / 16, mode=mode),
                  nn.Upsample(scale_factor=self.out_res / 32, mode=mode),
                  nn.Upsample(scale_factor=self.out_res / 32, mode=mode),
                  nn.Upsample(scale_factor=self.out_res / 64, mode=mode),
                  nn.Upsample(scale_factor=self.out_res / 64, mode=mode),
                  nn.Upsample(scale_factor=self.out_res / 128, mode=mode),
                  nn.Upsample(scale_factor=self.out_res / 128, mode=mode),
                  nn.Upsample(scale_factor=self.out_res / 256, mode=mode),
                  nn.Upsample(scale_factor=self.out_res / 256, mode=mode)]
        if self.out_res > 256:
            self.upsamplers.append(nn.Upsample(scale_factor=self.out_res / 512, mode=mode))
            self.upsamplers.append(nn.Upsample(scale_factor=self.out_res / 512, mode=mode))
        if model_path is not None:
            ckpt = torch.load(model_path, map_location='cpu')['model_state_dict']
            clean_ckpt = OrderedDict()
            for k in ckpt.keys():
                if k.startswith("module."):
                    clean_ckpt[k[7:]] = ckpt[k]
                else:
                    clean_ckpt[k] = ckpt[k]

            self.load_state_dict(clean_ckpt)

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, affine_layers):
        affine_layers_upsamples = []
        for i in range(len(self.upsamplers)):
            affine_layers_upsamples.append(self.upsamplers[i](
                affine_layers[i].detach()))
        feature_maps = torch.cat(affine_layers_upsamples, dim=1)

        feature_maps = feature_maps.permute(0, 2, 3, 1).contiguous()
        batch, h, w, dim = feature_maps.shape
        feature_maps = feature_maps.view(-1, dim)
        mask = self.layers(feature_maps)
        mask = torch.argmax(mask, dim=-1).view(batch, h, w)
        return mask.detach().cpu().numpy()
