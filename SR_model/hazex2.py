from turtle import forward
import torch
import torch.nn as nn
from model import common, dct
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import imageio
from torchvision import models

def make_model(opt):
    return HAZE_SR(opt)

# class hf_extractor(nn.Module):
#     def __init__(self, opt, conv=common.default_conv):
#         super(hf_extractor, self).__init__()
#         self.opt = opt
#         self.dct = dct.DCT_2D()
#         self.idct = dct.IDCT_2D()

#         _resnet = models.resnet18(pretrained=True)

#         self.head = conv(32, 3, 1)
        
#         self.resnet = nn.Sequential(
#             _resnet.conv1,
#             _resnet.bn1,
#             _resnet.relu,
#             _resnet.maxpool,
#             _resnet.layer1,
#             _resnet.layer2,
#             _resnet.layer3,
#             _resnet.layer4,
#             _resnet.avgpool
#         )

#         for param in self.resnet.parameters():
#             param.requires_grad = True

#         self.fc = nn.Sequential(
#             nn.Linear(512, 1),
#             nn.LeakyReLU()
#         )

#     def forward(self, x):

#         beta = self.head(x)
#         beta = self.resnet(beta)
#         beta = beta.view(beta.size(0), -1)
#         beta = self.fc(beta)
#         beta = torch.ones_like(beta) + beta.clone()

#         rec_beta = beta.clone()
#         _,_, h, w = x.size()
#         mask = torch.ones((h, w), dtype=torch.int64, device = torch.device('cuda:0'))
#         diagonal = 0

#         hf_mask = torch.fliplr(torch.triu(mask, diagonal)) != 1
#         lf_mask = torch.fliplr(torch.triu(mask, diagonal)) == 1

#         diagonal = -94
#         v_hf_mask = torch.fliplr(torch.triu(mask, diagonal)) != 1

#         hf_mask = hf_mask.bool().int()
#         v_hf_mask = v_hf_mask.bool().int()
#         hf_mask = hf_mask.unsqueeze(0).expand(x.size())
#         beta = beta.squeeze()[:, None, None, None]

#         hf_mask = hf_mask - v_hf_mask

#         hf = self.dct(x)
#         hf_mask = hf_mask * beta
#         hf_mask = hf_mask + lf_mask

#         hf = hf * hf_mask
#         hf = self.idct(hf)

#         return hf, rec_beta

class HAZE_SR(nn.Module):
    def __init__(self, opt, conv=common.default_conv):
        super(HAZE_SR, self).__init__()
        self.opt = opt
        self.scale = opt.scale
        self.phase = 1
        n_blocks = opt.n_blocks
        n_feats = 8
        self.n_hfab = 5
        kernel_size = 3
        # self.hf_extractor = hf_extractor(opt)

        act = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2,
                                    mode='bicubic', align_corners=False)

        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(opt.rgb_range, rgb_mean, rgb_std)

        self.head = conv(opt.n_colors, n_feats, kernel_size)

        self.down = [
            common.DownBlock(opt=opt, scale=2, nFeat=n_feats * pow(2, p), in_channels=n_feats, out_channels=n_feats * 4
            ) for p in range(self.phase)
        ]

        self.down = nn.ModuleList(self.down)

        self.rcab_group_0 = nn.ModuleList()
        for _ in range(self.n_hfab):
            self.rcab_group_0.append(
                nn.Sequential(*[
                common.RCAB(
                    conv, n_feats * 4, kernel_size, act=act
                ) for _ in range(n_blocks//self.n_hfab)
            ]))

        self.rcab_group_0 = nn.ModuleList(self.rcab_group_0)

        self.hfab_grop_0 = nn.ModuleList()
        for _ in range(self.n_hfab):
            self.hfab_grop_0.append(
                nn.Sequential(*[
                common.RCAB(
                    conv, n_feats * 4, kernel_size, act=act
                ) for _ in range(2)
            ]))

        self.hfab_grop_0 = nn.ModuleList(self.hfab_grop_0)


        self.rcab_group_1 = nn.ModuleList()
        for _ in range(self.n_hfab):
            self.rcab_group_1.append(
                nn.Sequential(*[
                common.RCAB(
                    conv, n_feats * 4, kernel_size, act=act
                ) for _ in range(n_blocks//self.n_hfab)
            ]))

        self.rcab_group_1 = nn.ModuleList(self.rcab_group_1)

        self.hfab_grop_1 = nn.ModuleList()
        for _ in range(self.n_hfab):
            self.hfab_grop_1.append(
                nn.Sequential(*[
                common.RCAB(
                    conv, n_feats * 4, kernel_size, act=act
                ) for _ in range(2)
            ]))

        self.hfab_grop_1 = nn.ModuleList(self.hfab_grop_1)

        # The fisrt upsample block
        up = [[
            common.Upsampler(conv, 2, n_feats * 4, act=False),
            conv(n_feats * 4, n_feats * pow(2, self.phase - 1), kernel_size=1)
        ]]

        # The rest upsample blocks
        for p in range(self.phase - 1, 0, -1):
            up.append([
                common.Upsampler(conv, 2, 2 * n_feats * pow(2, p), act=False),
                conv(2 * n_feats * pow(2, p), n_feats * pow(2, p - 1), kernel_size=1)
            ])

        self.up = nn.ModuleList()
        for idx in range(self.phase):
            self.up.append(
                nn.Sequential(*up[idx])
            )

        # tail conv that output sr imgs
        tail = [conv(n_feats * pow(2, self.phase), opt.n_colors, kernel_size)]
        for p in range(self.phase, 0, -1):
            tail.append(
                conv(n_feats*2, opt.n_colors, kernel_size)
            )
        self.tail = nn.ModuleList(tail)
        self.add_mean = common.MeanShift(opt.rgb_range, rgb_mean, rgb_std, 1)

        self.dct = dct.DCT_2D()
        self.idct = dct.IDCT_2D()


    def forward(self, x):
        # upsample x to target sr size
        x = self.upsample(x)

        # preprocess
        x = self.sub_mean(x)
        x = self.head(x)

        # down phases,
        copies = []
        for idx in range(self.phase):
            copies.append(x)
            x = self.down[idx](x)

        _,_, h, w = x.size()
        mask = torch.ones((h, w), dtype=torch.int64, device = torch.device('cuda:0'))
        diagonal = w-(w//6)
        hf_mask = torch.fliplr(torch.triu(mask, diagonal)) != 1
        hf_mask = hf_mask.unsqueeze(0).expand(x.size())

        for n in range(self.n_hfab):
            # hf, rec_beta = self.hf_extractor(x)
            hf = self.dct(x)
            hf = hf * hf_mask
            hf = self.idct(hf)
            hf = self.hfab_grop_0[n](hf)
            x = self.rcab_group_0[n](x)
            x = x + hf

        x = self.up[0](x)
        x = torch.cat((x, copies[0]), 1)

        # output sr imgss
        sr = self.tail[1](x)

        # sr = self.add_mean(sr)
        # results.append(sr)
        return sr