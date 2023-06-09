""" For searching the optimal set of regularization hyperparameters """
import glob
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as Data

from Functions import transform_unit_flow_to_flow_2D, \
    Dataset_epoch_validation
from miccai2021_model_2D import CSAIN_disp_lvl1, CSAIN_disp_lvl2, CSAIN_disp_lvl3, \
    SpatialTransform_unit, SpatialTransformNearest_unit, neg_Jdet_loss, DiceLoss, grad_Jdet_loss, JacboianDet

savepath = '../Result'
datapath = ''
modelpath = ''
if not os.path.isdir(savepath):
    os.mkdir(savepath)

imgshape = (160, 192)
imgshape_4 = (160 // 4, 192 // 4)
imgshape_2 = (160 // 2, 192 // 2)

range_flow = 0.4

start_channel = 64
iteration = 400
lr = 0.0003


def dice(im1, atlas):
    unique_class = np.unique(atlas)
    dice = []
    num_count = 0
    for i in unique_class:
        if (i == 0) or ((im1 == i).sum() == 0) or ((atlas == i).sum() == 0):
            continue

        sub_dice = np.sum(atlas[im1 == i] == i) * 2.0 / (np.sum(im1 == i) + np.sum(atlas == i))
        dice.append(sub_dice)
        # print(sub_dice)
    # print(num_count, len(unique_class)-1)
    return dice


def test():
    model_lvl1 = CSAIN_disp_lvl1(2, 2, start_channel, is_train=True,
                                                                         imgshape=imgshape_4,
                                                                         range_flow=range_flow).cuda()
    model_lvl2 = CSAIN_disp_lvl2(2, 2, start_channel, is_train=True,
                                                                         imgshape=imgshape_2,
                                                                         range_flow=range_flow,
                                                                         model_lvl1=model_lvl1).cuda()

    model = CSAIN_disp_lvl3(2, 2, start_channel, is_train=False,
                                                                    imgshape=imgshape,
                                                                    range_flow=range_flow, model_lvl2=model_lvl2).cuda()

    transform = SpatialTransform_unit().cuda()
    transform_nearest = SpatialTransformNearest_unit().cuda()

    model.load_state_dict(torch.load(modelpath))
    model.eval()

    grid = F.affine_grid(torch.eye(3)[0:2].unsqueeze(0), (1,) + (1,) + imgshape, align_corners=True).cuda()

    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    my_dice = DiceLoss()

    imgs = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/slice_norm.nii.gz'))[-10:]
    labels = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/slice_seg24.nii.gz'))[-10:]
    seg_name = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/slice_seg4.nii.gz'))[-10:]

    valid_generator = Data.DataLoader(
        Dataset_epoch_validation(imgs, labels, seg_name, norm=True),
        batch_size=5,
        shuffle=False, num_workers=2)

    step = 0

    loss_smooth = neg_Jdet_loss

    dice_total = []
    Jacobian_total = []
    grad_total = []
    reg_total = []
    std_total = []

    last_loss = torch.Tensor(0).cuda()

    for batch_idx, data in enumerate(valid_generator):
        X, Y, X_label, Y_label, seg_Y = data[0].squeeze(-1).to(device), data[1].squeeze(-1).to(device), \
                                        data[
                                            2].squeeze(-1).to(device), data[3].squeeze(-1).to(device), data[
                                            4].squeeze(-1).to(device)

        reg_input = torch.tensor([0.5, 0.3, 0.3, 0.3, 0.3], dtype=Y.dtype, device=Y.device)
        reg_code = torch.nn.Parameter(data=reg_input, requires_grad=True)

        optimizer = torch.optim.Adam([reg_code], lr=lr)

        while step <= iteration:
            F_X_Y = model(X, Y, seg_Y, reg_code)

            X_Y = transform(X, F_X_Y.permute(0, 2, 3, 1), grid)

            # aligned_label = transform(X_label, F_X_Y.permute(0, 2, 3, 1), grid)

            loss_dice = my_dice(X_Y, Y)

            F_X_Y_norm = transform_unit_flow_to_flow_2D(F_X_Y.permute(0, 2, 3, 1))

            _, _, x, y = F_X_Y.shape
            norm_vector = torch.zeros((1, 2, 1, 1), dtype=F_X_Y.dtype, device=F_X_Y.device)
            norm_vector[0, 0, 0, 0] = (y - 1)
            norm_vector[0, 1, 0, 0] = (x - 1)

            # loss_regulation = loss_smooth(F_X_Y_norm, grid)

            F_X_Y_cpu = F_X_Y.unsqueeze(-1).data.cpu().numpy()[0, :, :, :, :].transpose(1, 2, 3, 0)
            x, y, z, _ = F_X_Y_cpu.shape
            F_X_Y_cpu[:, :, :, 0] = F_X_Y_cpu[:, :, :, 0] * (y - 1) / 2
            F_X_Y_cpu[:, :, :, 1] = F_X_Y_cpu[:, :, :, 1] * (x - 1) / 2

            loss = loss_dice

            loss_diff = torch.abs(loss - last_loss).detach().cpu().numpy()
            if loss_diff < 0.000004:
                break

            last_loss = loss

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            step += 1

        step = 0

        F_X_Y = model(X, Y, seg_Y, reg_code)

        aligned_label = transform_nearest(X_label, F_X_Y.permute(0, 2, 3, 1), grid).detach().cpu().numpy()[0, 0, :, :]

        fixed_label_img = Y_label.cpu().numpy()[0, 0, :, :]

        F_X_Y_norm = transform_unit_flow_to_flow_2D(F_X_Y.permute(0, 2, 3, 1).clone())

        loss_Jacobian = neg_Jdet_loss(F_X_Y_norm, grid).detach().cpu().numpy()

        F_X_Y_cpu = F_X_Y.unsqueeze(-1).data.cpu().numpy()[0, :, :, :, :].transpose(1, 2, 3, 0)
        x, y, z, _ = F_X_Y_cpu.shape
        F_X_Y_cpu[:, :, :, 0] = F_X_Y_cpu[:, :, :, 0] * (y - 1) / 2
        F_X_Y_cpu[:, :, :, 1] = F_X_Y_cpu[:, :, :, 1] * (x - 1) / 2

        dice_score = dice(np.floor(aligned_label), np.floor(fixed_label_img))

        grad_Jaco = grad_Jdet_loss(F_X_Y_norm, grid).detach().cpu().numpy()

        Jaco = JacboianDet(F_X_Y_norm, grid).detach().cpu().numpy()

        dice_total.append(dice_score)
        Jacobian_total.append(loss_Jacobian)
        grad_total.append(grad_Jaco)
        reg_total.append(reg_code.detach().cpu().numpy())
        std_total.append(np.std(Jaco))

    dic = pd.DataFrame(dice_total)
    dic.to_csv('dice.csv')
    Jac = pd.DataFrame(Jacobian_total)
    Jac.to_csv('Jacobian.csv')
    grad = pd.DataFrame(grad_total)
    grad.to_csv('grad.csv')
    reg = pd.DataFrame(reg_total)
    reg.to_csv('reg.csv')
    std = pd.DataFrame(std_total)
    std.to_csv('std.csv')


if __name__ == '__main__':
    imgshape = (160, 192)
    imgshape_4 = (160 // 4, 192 // 4)
    imgshape_2 = (160 // 2, 192 // 2)

    range_flow = 0.4
    test()
