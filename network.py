import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
from matplotlib import colors


class CSAIN_disp_lvl1(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, imgshape=(160, 192), range_flow=0.4):
        super(CSAIN_disp_lvl1, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.imgshape = imgshape
        self.grid_1 = F.affine_grid(torch.eye(3)[0:2].unsqueeze(0), (1,) + (1,) + self.imgshape,
                                    align_corners=True).cuda()

        self.transform = SpatialTransform_unit().cuda()

        self.gaussblur = torchvision.transforms.GaussianBlur(3, 0.5)

        bias_opt = False

        self.input_encoder_lvl1 = self.input_feature_extract(self.in_channel, self.start_channel * 4, bias=bias_opt)

        self.down_conv = nn.Conv2d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1,
                                   bias=bias_opt)

        self.reg_encoder = self.reg_conv(self.in_channel, self.start_channel * 2, bias=bias_opt)

        self.resblock_group_lvl1 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        self.up = nn.ConvTranspose2d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        self.down_avg = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        self.output_lvl1 = self.outputs(self.start_channel * 8, self.n_classes, kernel_size=3, stride=1, padding=1,
                                        bias=False)

    def resblock_seq(self, in_channels, bias_opt=False):
        layer = nn.ModuleList([
            PreActBlock_CSAIN(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock_CSAIN(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock_CSAIN(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock_CSAIN(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock_CSAIN(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2)
        ]
        )
        return layer

    def input_feature_extract(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                              bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer

    def reg_conv(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):

        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU())

        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.Tanh())
        else:
            # layer = nn.Sequential(
            #     nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            #     nn.Tanh())
            layer = nn.Sequential(
                nn.Conv2d(in_channels, int(in_channels / 2), kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv2d(int(in_channels / 2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, x, y, seg_Y, reg_code):

        cat_input = torch.cat((x, y), 1)
        cat_input = self.down_avg(cat_input)
        cat_input_lvl1 = self.down_avg(cat_input)

        seg_Y_lv2 = self.down_avg(seg_Y)
        seg_Y_lv1 = self.down_avg(seg_Y_lv2)

        reg_matrix = regularization_matrix(seg_Y_lv1, reg_code)
        # reg_matrix = self.gaussblur(reg_matrix) for generating gaussian blurred regularization matrix

        down_y = cat_input_lvl1[:, 1:2, :, :]

        fea_e0 = self.input_encoder_lvl1(cat_input_lvl1)
        e0 = self.down_conv(fea_e0)

        fea_reg_matrix = F.interpolate(reg_matrix, scale_factor=0.5)
        fea_reg_matrix = self.reg_encoder(fea_reg_matrix)

        # e0 = self.resblock_group_lvl1(e0)
        for i in range(len(self.resblock_group_lvl1)):
            if i % 2 == 0:
                e0 = self.resblock_group_lvl1[i](e0, fea_reg_matrix)
            else:
                e0 = self.resblock_group_lvl1[i](e0)

        e0 = self.up(e0)
        output_disp_e0_v = self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        warpped_inputx_lvl1_out = self.transform(x, output_disp_e0_v.permute(0, 2, 3, 1), self.grid_1)

        if self.is_train is True:
            return output_disp_e0_v, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0, reg_matrix
        else:
            return output_disp_e0_v


class CSAIN_disp_lvl2(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, imgshape=(160, 192), range_flow=0.4,
                 model_lvl1=None):
        super(CSAIN_disp_lvl2, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.imgshape = imgshape

        self.model_lvl1 = model_lvl1

        self.grid_1 = F.affine_grid(torch.eye(3)[0:2].unsqueeze(0), (1,) + (1,) + self.imgshape,
                                    align_corners=True).cuda()

        self.transform = SpatialTransform_unit().cuda()

        self.gaussblur = torchvision.transforms.GaussianBlur(5, 0.4)

        bias_opt = False

        self.input_encoder_lvl1 = self.input_feature_extract(self.in_channel + 2, self.start_channel * 4, bias=bias_opt)

        self.down_conv = nn.Conv2d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1,
                                   bias=bias_opt)

        self.reg_encoder = self.reg_conv(self.in_channel, self.start_channel * 2, bias=bias_opt)

        self.resblock_group_lvl1 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="bilinear")
        self.up = nn.ConvTranspose2d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        self.down_avg = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)

        self.output_lvl1 = self.outputs(self.start_channel * 8, self.n_classes, kernel_size=3, stride=1, padding=1,
                                        bias=False)

    def unfreeze_modellvl1(self):
        # unFreeze model_lvl1 weight
        print("\nunfreeze model_lvl1 parameter")
        for param in self.model_lvl1.parameters():
            param.requires_grad = True

    def resblock_seq(self, in_channels, bias_opt=False):
        layer = nn.ModuleList([
            PreActBlock_CSAIN(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock_CSAIN(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock_CSAIN(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock_CSAIN(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock_CSAIN(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2)
        ]
        )
        return layer

    def input_feature_extract(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                              bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer

    def reg_conv(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):

        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.ReLU())

        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.Tanh())
        else:
            # layer = nn.Sequential(
            #     nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            #     nn.Tanh())
            layer = nn.Sequential(
                nn.Conv2d(in_channels, int(in_channels / 2), kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv2d(int(in_channels / 2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, x, y, seg_Y, reg_code):
        # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0
        lvl1_disp, _, _, lvl1_v, lvl1_embedding, _ = self.model_lvl1(x, y, seg_Y, reg_code)
        lvl1_disp_up = self.up_tri(lvl1_disp)

        x_down = self.down_avg(x)
        y_down = self.down_avg(y)
        seg_Y_lv2 = self.down_avg(seg_Y)
        reg_matrix = regularization_matrix(seg_Y_lv2, reg_code)
        # reg_matrix = self.gaussblur(reg_matrix) for generating gaussian blurred regularization matrix

        warpped_x = self.transform(x_down, lvl1_disp_up.permute(0, 2, 3, 1), self.grid_1)

        cat_input_lvl2 = torch.cat((warpped_x, y_down, lvl1_disp_up), 1)

        fea_e0 = self.input_encoder_lvl1(cat_input_lvl2)
        e0 = self.down_conv(fea_e0)

        fea_reg_matrix = F.interpolate(reg_matrix, scale_factor=0.5)
        fea_reg_matrix = self.reg_encoder(fea_reg_matrix)

        e0 = e0 + lvl1_embedding

        # e0 = self.resblock_group_lvl1(e0)
        for i in range(len(self.resblock_group_lvl1)):
            if i % 2 == 0:
                e0 = self.resblock_group_lvl1[i](e0, fea_reg_matrix)
            else:
                e0 = self.resblock_group_lvl1[i](e0)

        e0 = self.up(e0)
        output_disp_e0_v = self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        compose_field_e0_lvl1 = lvl1_disp_up + output_disp_e0_v
        warpped_inputx_lvl1_out = self.transform(x, compose_field_e0_lvl1.permute(0, 2, 3, 1), self.grid_1)

        if self.is_train is True:
            return compose_field_e0_lvl1, warpped_inputx_lvl1_out, y_down, output_disp_e0_v, lvl1_v, e0, reg_matrix
        else:
            return compose_field_e0_lvl1


class CSAIN_disp_lvl3(nn.Module):
    def __init__(self, in_channel, n_classes, start_channel, is_train=True, imgshape=(160, 192), range_flow=0.4,
                 model_lvl2=None):
        super(CSAIN_disp_lvl3, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.start_channel = start_channel

        self.range_flow = range_flow
        self.is_train = is_train

        self.imgshape = imgshape

        self.model_lvl2 = model_lvl2

        self.grid_1 = F.affine_grid(torch.eye(3)[0:2].unsqueeze(0), (1,) + (1,) + self.imgshape,
                                    align_corners=True).cuda()

        self.transform = SpatialTransform_unit().cuda()

        self.gaussblur = torchvision.transforms.GaussianBlur(5, 0.8)

        bias_opt = False

        self.input_encoder_lvl1 = self.input_feature_extract(self.in_channel + 2, self.start_channel * 4, bias=bias_opt)

        self.down_conv = nn.Conv2d(self.start_channel * 4, self.start_channel * 4, 3, stride=2, padding=1,
                                   bias=bias_opt)

        self.reg_encoder = self.reg_conv(self.in_channel, self.start_channel * 2, bias=bias_opt)

        self.resblock_group_lvl1 = self.resblock_seq(self.start_channel * 4, bias_opt=bias_opt)

        self.up_tri = torch.nn.Upsample(scale_factor=2, mode="bilinear")
        self.up = nn.ConvTranspose2d(self.start_channel * 4, self.start_channel * 4, 2, stride=2,
                                     padding=0, output_padding=0, bias=bias_opt)

        self.output_lvl1 = self.outputs(self.start_channel * 8, self.n_classes, kernel_size=3, stride=1, padding=1,
                                        bias=False)

    def unfreeze_modellvl2(self):
        # unFreeze model_lvl1 weight
        print("\nunfreeze model_lvl2 parameter")
        for param in self.model_lvl2.parameters():
            param.requires_grad = True

    def resblock_seq(self, in_channels, bias_opt=False):
        layer = nn.ModuleList([
            PreActBlock_CSAIN(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock_CSAIN(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock_CSAIN(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock_CSAIN(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2),
            PreActBlock_CSAIN(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2)
        ]
        )
        return layer

    def input_feature_extract(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                              bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias))
        return layer

    def reg_conv(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):

        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.LeakyReLU(0.2))

        return layer

    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.ReLU())
        return layer

    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, int(in_channels / 2), kernel_size, stride=stride, padding=padding, bias=bias),
                nn.LeakyReLU(0.2),
                nn.Conv2d(int(in_channels / 2), out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer

    def forward(self, x, y, seg_Y, reg_code):
        lvl2_disp, _, _, lvl2_v, lvl1_v, lvl2_embedding, _ = self.model_lvl2(x, y, seg_Y, reg_code)
        lvl2_disp_up = self.up_tri(lvl2_disp)
        warpped_x = self.transform(x, lvl2_disp_up.permute(0, 2, 3, 1), self.grid_1)

        cat_input = torch.cat((warpped_x, y, lvl2_disp_up), 1)

        reg_matrix = regularization_matrix(seg_Y, reg_code)
        # reg_matrix = self.gaussblur(reg_matrix)
        plt.imshow(1 - reg_matrix[0, 0, :, :].detach().cpu().numpy(), cmap='bone_r')
        plt.show()

        fea_e0 = self.input_encoder_lvl1(cat_input)
        e0 = self.down_conv(fea_e0)

        fea_reg_matrix = F.interpolate(reg_matrix, scale_factor=0.5)
        fea_reg_matrix = self.reg_encoder(fea_reg_matrix)

        e0 = e0 + lvl2_embedding

        for i in range(len(self.resblock_group_lvl1)):
            if i % 2 == 0:
                e0 = self.resblock_group_lvl1[i](e0, fea_reg_matrix)
            else:
                e0 = self.resblock_group_lvl1[i](e0)

        e0 = self.up(e0)
        output_disp_e0_v = self.output_lvl1(torch.cat([e0, fea_e0], dim=1)) * self.range_flow
        compose_field_e0_lvl1 = output_disp_e0_v + lvl2_disp_up

        warpped_inputx_lvl1_out = self.transform(x, compose_field_e0_lvl1.permute(0, 2, 3, 1), self.grid_1)

        if self.is_train is True:
            return compose_field_e0_lvl1, warpped_inputx_lvl1_out, y, output_disp_e0_v, lvl1_v, lvl2_v, e0, reg_matrix
        else:
            return compose_field_e0_lvl1


class CSAIN(nn.Module):
    """Conditional Spatially-Adaptive Instance Normalization (CSAIN)"""
    def __init__(self, in_channel, in_channels=128, out_channels=256):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)

        self.gamma_gen = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        )

        self.beta_gen = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2)
        )

    def forward(self, input, reg_matrix):
        # style [batch_size, in_channels*2] => [batch_size, in_channels*2, 1, 1, 1]
        gamma = self.gamma_gen(reg_matrix)
        beta = self.beta_gen(reg_matrix)
        # sns.set_context({"figure.figsize": (8, 8)})
        # sns.heatmap(data=gamma[0, 128, :, :].cpu().numpy(), square=True)
        # plt.show()

        out = self.norm(input)
        # out = input
        out = (1. + gamma) * out + beta

        return out


class PreActBlock_CSAIN(nn.Module):
    """Pre-activation version of the BasicBlock + Conditional Spatially Adaptive Instance Normalization"""
    expansion = 1

    def __init__(self, in_planes, planes, num_group=4, stride=1, bias=False, latent_dim=64, mapping_fmaps=64):
        super(PreActBlock_CSAIN, self).__init__()
        self.ai1 = CSAIN(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.ai2 = CSAIN(in_planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=bias)
            )

    def forward(self, x, reg_matrix):
        out = F.leaky_relu(self.ai1(x, reg_matrix), negative_slope=0.2)

        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)

        out = self.conv2(F.leaky_relu(self.ai2(out, reg_matrix), negative_slope=0.2))

        out += shortcut
        return out


class SpatialTransform_unit(nn.Module):
    def __init__(self):
        super(SpatialTransform_unit, self).__init__()

    def forward(self, x, flow, sample_grid):
        sample_grid = sample_grid + flow
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='bilinear', padding_mode="border",
                                               align_corners=True)

        return flow


class SpatialTransformNearest_unit(nn.Module):
    def __init__(self):
        super(SpatialTransformNearest_unit, self).__init__()

    def forward(self, x, flow, sample_grid):
        sample_grid = sample_grid + flow
        df = pd.DataFrame(sample_grid[0, :, :, 0].detach().cpu().numpy())
        df.to_csv('grid.csv')
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='nearest', padding_mode="border",
                                               align_corners=True)

        return flow


def spatially_variant_reg(y_pred, reg_matrix):
    """Spatially variant regularization"""
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
    _, _, x, y = reg_matrix.shape
    regx_matrix = reg_matrix[:, :, :, 0:(y - 1)]
    regy_matrix = reg_matrix[:, :, 0:(x - 1), :]
    reg_dx = regx_matrix * dx
    reg_dy = regy_matrix * dy

    return (torch.mean(reg_dx * reg_dx) + torch.mean(reg_dy * reg_dy)) / 2.0


def regularization_matrix(F_seg, reg_code):
    """Regularization matrix for the spatially variant regularization"""
    b, _, x, y = F_seg.shape
    reg_matrix = torch.FloatTensor(b, 2, x, y).cuda()
    F_seg = (F_seg * torch.tensor(4).cuda()).round()
    F_seg = F_seg * torch.tensor(2).cuda()
    df = F_seg[:, 0, :, :]
    df[df == 0] = reg_code[0]
    df[df == 2] = reg_code[1]
    df[df == 4] = reg_code[2]
    df[df == 6] = reg_code[3]
    df[df == 8] = reg_code[4]
    reg_matrix[:, 0, :, :] = df
    reg_matrix[:, 1, :, :] = reg_matrix[:, 0, :, :]

    return reg_matrix


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


def JacboianDet(y_pred, sample_grid):
    J = y_pred + sample_grid
    dy = J[:, 1:, :-1, :] - J[:, :-1, :-1, :]
    dx = J[:, :-1, 1:, :] - J[:, :-1, :-1, :]

    Jdet = dx[:, :, :, 0] * dy[:, :, :, 1] - dx[:, :, :, 1] * dy[:, :, :, 0]

    return Jdet


def neg_Jdet_loss(y_pred, sample_grid):
    neg_Jdet = -1.0 * JacboianDet(y_pred, sample_grid)
    selected_neg_Jdet = F.relu(neg_Jdet)
    non_zero = torch.norm(selected_neg_Jdet, p=0) / (159 * 161)

    return non_zero


def neg_Jdet_mean(y_pred, sample_grid):
    neg_Jdet = -1.0 * JacboianDet(y_pred, sample_grid)
    selected_neg_Jdet = F.relu(neg_Jdet)

    return torch.mean(selected_neg_Jdet)


def grad_Jdet_loss(y_pred, sample_grid):
    J_det = JacboianDet(y_pred, sample_grid)
    dy = torch.abs(J_det[:, 1:, :] - J_det[:, :-1, :])
    dx = torch.abs(J_det[:, :, 1:] - J_det[:, :, :-1])
    return torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))


def par_Jdet_loss(y_pred, sample_grid, count):
    neg_Jdet = -1.0 * JacboianDet(y_pred, sample_grid)
    selected_neg_Jdet = F.relu(neg_Jdet)

    # return selected_neg_Jdet
    return torch.sum(selected_neg_Jdet) / count


def mse_loss(input, target):
    y_true_f = input.view(-1)
    y_pred_f = target.view(-1)
    diff = y_true_f - y_pred_f
    mse = torch.mul(diff, diff).mean()
    return mse


class NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """

    def __init__(self, win=5, eps=1e-8):
        super(NCC, self).__init__()
        self.win = win
        self.eps = eps
        self.w_temp = win

    def forward(self, I, J):
        ndims = 2
        win_size = self.w_temp

        # set window size
        if self.win is None:
            self.win = [5] * ndims
        else:
            self.win = [self.w_temp] * ndims

        weight_win_size = self.w_temp
        weight = torch.ones((1, 1, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        conv_fn = F.conv2d

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size / 2))
        J_sum = conv_fn(J, weight, padding=int(win_size / 2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size / 2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size / 2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size / 2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)


class multi_resolution_NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """

    def __init__(self, win=None, eps=1e-5, scale=3):
        super(multi_resolution_NCC, self).__init__()
        self.num_scale = scale
        # self.similarity_metric = NCC(win=win)

        self.similarity_metric = []

        for i in range(scale):
            self.similarity_metric.append(NCC(win=win - (i * 2)))
            # self.similarity_metric.append(Normalized_Gradient_Field(eps=0.01))

    def forward(self, I, J):
        total_NCC = []
        # scale_I = I
        # scale_J = J
        #
        # for i in range(self.num_scale):
        #     current_NCC = similarity_metric(scale_I,scale_J)
        #     # print("Scale ", i, ": ", current_NCC, (2**i))
        #     total_NCC += current_NCC/(2**i)
        #     # print(scale_I.size(), scale_J.size())
        #     # print(current_NCC)
        #     scale_I = nn.functional.interpolate(I, scale_factor=(1.0/(2**(i+1))))
        #     scale_J = nn.functional.interpolate(J, scale_factor=(1.0/(2**(i+1))))

        for i in range(self.num_scale):
            current_NCC = self.similarity_metric[i](I, J)
            total_NCC.append(current_NCC / (2 ** i))
            # print(scale_I.size(), scale_J.size())

            I = nn.functional.avg_pool2d(I, kernel_size=3, stride=2, padding=1, count_include_pad=False)
            J = nn.functional.avg_pool2d(J, kernel_size=3, stride=2, padding=1, count_include_pad=False)

        return sum(total_NCC)


def dice_coeff(probabilities, labels):
    """Compute a mean hard or soft dice coefficient between a batch of probabilities and target
    labels. Reduction happens over the batch dimension; if None, return dice per example.
    """
    # This factor prevents division by 0 if both prediction and GT don't have any foreground voxels
    smooth = 1e-3

    # Flatten all dims except for the batch
    probabilities_flat = torch.flatten(probabilities, start_dim=1)
    labels_flat = torch.flatten(labels, start_dim=1)

    intersection = (probabilities_flat * labels_flat).sum(dim=1)
    volume_sum = probabilities_flat.sum(dim=1) + labels_flat.sum(dim=1)  # it's not the union!
    dice = (intersection + smooth) / (volume_sum + smooth)
    dice = torch.mean(dice)

    return dice


class DiceLoss(torch.nn.Module):
    """Takes logits as input."""

    def __init__(self, smooth=1e-3, ):
        super(DiceLoss, self).__init__()

        self.smooth = smooth

    def forward(self, y_pred, y_true):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)

        return -dice


class MSELoss(torch.nn.Module):
    """Takes logits as input."""

    def __init__(self, smooth=1e-3, ):
        super(MSELoss, self).__init__()

        self.smooth = smooth

    def forward(self, y_pred, y_true):
        error = y_pred - y_true
        MSE = torch.mean(error * error)

        return MSE
