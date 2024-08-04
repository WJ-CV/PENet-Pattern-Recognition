import torch
from torch import nn
import torch.nn.functional as F
import cv2
import pvt_v2
from torch.nn import Conv2d, Parameter, Softmax
import numpy as np
import os

def convblock(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )
def convblock2(in_ch, out_ch, rate):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, dilation=rate, padding=rate),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="bchw"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["bchw", "bhwc"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "bhwc":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "bchw":
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
class CA(nn.Module):
    def __init__(self,in_ch):
        super(CA, self).__init__()
        self.avg_weight = nn.AdaptiveAvgPool2d(1)
        self.max_weight = nn.AdaptiveMaxPool2d(1)
        self.fus = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(in_ch // 2, in_ch, 1, 1, 0),
        )
        self.c_mask = nn.Sigmoid()
    def forward(self, x):
        avg_map_c = self.avg_weight(x)
        max_map_c = self.max_weight(x)
        c_mask = self.c_mask(torch.add(self.fus(avg_map_c), self.fus(max_map_c)))
        return torch.mul(x, c_mask)

class GFAPF(nn.Module):
    def __init__(self, dim):
        super(GFAPF, self).__init__()
        self.conv_r = nn.Conv2d(4*dim, dim // 4, 1, 1, 0)
        self.act = nn.GELU()
        self.lka1 = nn.Sequential(
            nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4),
            nn.Conv2d(dim // 4, dim // 4, 3, stride=1, padding=1, groups=dim // 4, dilation=1),
            nn.Conv2d(dim // 4, dim // 4, 1),
        )
        self.lka2 = nn.Sequential(
            nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4),
            nn.Conv2d(dim // 4, dim // 4, 3, stride=1, padding=2, groups=dim // 4, dilation=2),
            nn.Conv2d(dim // 4, dim // 4, 1),
        )
        self.lka3 = nn.Sequential(
            nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4),
            nn.Conv2d(dim // 4, dim // 4, 3, stride=1, padding=4, groups=dim // 4, dilation=4),
            nn.Conv2d(dim // 4, dim // 4, 1),
        )
        self.lka4 = nn.Sequential(
            nn.Conv2d(dim // 4, dim // 4, 3, padding=1, groups=dim // 4),
            nn.Conv2d(dim // 4, dim // 4, 3, stride=1, padding=6, groups=dim // 4, dilation=6),
            nn.Conv2d(dim // 4, dim // 4, 1),
        )
        self.fus = convblock(dim, dim, 1, 1, 0)

    def forward(self, rgb, t):
        x = torch.cat((torch.cat((rgb, t), 1), 0.5 * (rgb + t), rgb * t), 1)
        in_x = self.act(self.conv_r(x))
        x1 = self.lka1(in_x)
        x2 = self.lka2(in_x)
        x3 = self.lka3(in_x)
        x4 = self.lka4(in_x)

        out = self.fus(torch.cat((x1, x2, x3, x4), 1))
        return out

class GMLP(nn.Module):
    def __init__(self, dim):
        super(GMLP, self).__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim // 4)
        self.fc1 = nn.Linear(dim, dim // 4)
        self.fc3 = nn.Linear(dim // 4, dim)
        self.act = nn.GELU()
        self.conv1d = nn.Conv1d(dim // 4, dim // 4, 1, 1, int((1 - 1) / 2))
        self.conv3d = nn.Conv1d(dim // 4, dim // 4, 3, 1, int((3 - 1) / 2))
        self.conv5d = nn.Conv1d(dim // 4, dim // 4, 5, 1, int((5 - 1) / 2))
        self.conv9d = nn.Conv1d(dim // 4, dim // 4, 9, 1, int((9 - 1) / 2))
        self.dw = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0, groups=dim),
            nn.Conv2d(dim, dim, 1),
        )
        nn.init.constant_(self.conv1d.bias, 1.0)
        self.conv_fus = convblock(4 * dim, dim, 3, 1, 1)

    def forward(self, rgb0, t0):
        b, c, h, w = rgb0.size()
        rgb = self.ln1(rgb0.view(b, c, -1).permute(0, 2, 1))  #  b, l, c
        t = self.ln1(t0.view(b, c, -1).permute(0, 2, 1))
        rgb = self.ln2(self.act(self.fc1(rgb))).permute(0, 2, 1)   #  b, c, l
        t = self.ln2(self.act(self.fc1(t))).permute(0, 2, 1)

        r1 = self.conv1d(rgb)
        t1 = self.conv1d(t)
        rt1 = (r1 * t1).permute(0, 2, 1)
        rt1 = self.dw(self.fc3(rt1).permute(0, 2, 1).view(b, c, h, w))

        r3 = self.conv3d(rgb)
        t3 = self.conv3d(t)
        rt3 = (r3 * t3).permute(0, 2, 1)
        rt3 = self.dw(self.fc3(rt3).permute(0, 2, 1).view(b, c, h, w))

        r5 = self.conv5d(rgb)
        t5 = self.conv5d(t)
        rt5 = (r5 * t5).permute(0, 2, 1)
        rt5 = self.dw(self.fc3(rt5).permute(0, 2, 1).view(b, c, h, w))

        r9 = self.conv9d(rgb)
        t9 = self.conv9d(t)
        rt9 = (r9 * t9).permute(0, 2, 1)
        rt9 = self.dw(self.fc3(rt9).permute(0, 2, 1).view(b, c, h, w))

        rt_out = torch.cat((rt3, rt1, rt5, rt9), 1)

        return self.conv_fus(rt_out)
    
class HMF(nn.Module):
    def __init__(self, in_1, in_2):
        super(HMF, self).__init__()
        self.gmlp = GMLP(in_2)
        self.casa1 = CA(in_2)
        self.conv_m = convblock(in_2, in_2, 3, 1, 1)
        self.conv_globalinfo = convblock(in_1, in_2, 3, 1, 1)
        self.conv_out = convblock(2*in_2, in_2, 1, 1, 0)
        self.rt_fus = nn.Sequential(
            nn.Conv2d(in_2, in_2, 3, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, cu_p, cu_s, global_info):
        cur_size = cu_p.size()[2:]
        cu_p = self.casa1(cu_p)
        cu_s = self.casa1(cu_s)
        cross_cat = self.gmlp(cu_p, cu_s)
        cm_fus = self.conv_m(cross_cat)
        global_info = self.conv_globalinfo(F.interpolate(global_info, cur_size, mode='bilinear', align_corners=True))

        # cross_cat = cm_fus + torch.add(cm_fus, torch.mul(cm_fus, self.rt_fus(global_info)))
        global_info = global_info + torch.add(global_info, torch.mul(global_info, self.rt_fus(cm_fus)))
        return self.conv_out(torch.cat((cm_fus, global_info), 1))
    
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.glo = GFAPF(512)
        self.fus3 = HMF(512, 512)
        self.fus2 = HMF(512, 320)
        self.fus1 = HMF(320, 128)
        self.fus0 = HMF(128, 64)
        self.sig = nn.Sigmoid()
        self.up_4 = convblock(512, 512, 1, 1, 0)
        self.up_3 = convblock(512, 320, 3, 1, 1)
        self.up_2 = convblock(320, 128, 3, 1, 1)
        self.up_1 = convblock(128, 64, 3, 1, 1)

        self.conv3 = convblock(320, 64, 3, 1, 1)
        self.conv2 = convblock(128, 64, 3, 1, 1)
        self.conv1 = convblock(64, 64, 1, 1, 0)
        self.conv0 = convblock(64, 64, 1, 1, 0)

        self.conv_up1 = convblock(64 * 3, 64, 3, 1, 1)

        self.conv_up2 = convblock(64 * 3, 64, 3, 1, 1)

    def forward(self, rgb_f, t_f):

        fus_glo = self.glo(rgb_f[3], t_f[3])  # 768
        fus_3 = self.fus3(rgb_f[3], t_f[3], fus_glo)
        fus_2 = self.fus2(rgb_f[2], t_f[2], fus_3)
        fus_1 = self.fus1(rgb_f[1], t_f[1], fus_2)
        fus_0 = self.fus0(rgb_f[0], t_f[0], fus_1)


        s3 = fus_3 + torch.mul(fus_3, self.sig(self.up_4(fus_glo)))
        up_3 = self.up_3(F.interpolate(s3, fus_2.size()[2:], mode='bilinear', align_corners=True))
        s2 = fus_2 + torch.mul(fus_2, self.sig(up_3))
        up_2 = self.up_2(F.interpolate(s2, fus_1.size()[2:], mode='bilinear', align_corners=True))
        s1 = fus_1 + torch.mul(fus_1, self.sig(up_2))
        up_1 = self.up_1(F.interpolate(s1, fus_0.size()[2:], mode='bilinear', align_corners=True))
        s0 = fus_0 + torch.mul(fus_0, self.sig(up_1))

        d3_up = self.conv3(F.interpolate(up_3, s0.size()[2:], mode='bilinear', align_corners=True))
        d2_up = self.conv2(F.interpolate(up_2, s0.size()[2:], mode='bilinear', align_corners=True))
        d1_up = self.conv1(up_1)
        d0_up = self.conv0(s0)

        up1 = self.conv_up1(torch.cat((d3_up, d2_up, d1_up), 1))
        up2 = self.conv_up2(torch.cat((up1, d1_up, d0_up), 1))

        return up1, up2

class Transformer(nn.Module):
    def __init__(self, backbone, pretrained=None):
        super().__init__()
        self.encoder = getattr(pvt_v2, backbone)()
        if pretrained:
            checkpoint = torch.load('../pvt_v2_b3.pth', map_location='cpu')
            if 'model' in checkpoint:
                checkpoint_model = checkpoint['model']
            else:
                checkpoint_model = checkpoint
            state_dict = self.encoder.state_dict()
            for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            self.encoder.load_state_dict(checkpoint_model, strict=False)

def Encoder():
    model = Transformer('pvt_v2_b3', pretrained=True)
    return model

class Mnet(nn.Module):
    def __init__(self):
        super(Mnet, self).__init__()
        model = Encoder()
        self.rgb_net = model.encoder
        self.t_net = model.encoder
        self.decoder = Decoder()

    def forward(self, rgb, t):
        rgb_f = []
        t_f = []
        rgb_f = self.rgb_net(rgb)
        t_f = self.t_net(t)

        up1, up2 = self.decoder(rgb_f, t_f)

        return up1, up2