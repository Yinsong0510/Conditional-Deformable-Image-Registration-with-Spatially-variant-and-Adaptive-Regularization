import glob
import os
import sys
from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data

from Functions import Dataset_epoch, Dataset_epoch_validation, transform_unit_flow_to_flow_2D
from miccai2021_model_2D import CSAIN_disp_lvl1, \
    CSAIN_disp_lvl2, CSAIN_disp_lvl3, \
    SpatialTransform_unit, SpatialTransformNearest_unit, spatially_variant_reg, \
    NCC, multi_resolution_NCC, neg_Jdet_loss

parser = ArgumentParser()
parser.add_argument("--lr", type=float,
                    dest="lr", default=1e-4, help="learning rate")
parser.add_argument("--iteration_lvl1", type=int,
                    dest="iteration_lvl1", default=120001,
                    help="number of lvl1 iterations")
parser.add_argument("--iteration_lvl2", type=int,
                    dest="iteration_lvl2", default=160001,
                    help="number of lvl2 iterations")
parser.add_argument("--iteration_lvl3", type=int,
                    dest="iteration_lvl3", default=240001,
                    help="number of lvl3 iterations")
parser.add_argument("--antifold", type=float,
                    dest="antifold", default=0.,
                    help="Anti-fold loss: Disabled")
parser.add_argument("--checkpoint", type=int,
                    dest="checkpoint", default=5000,
                    help="frequency of saving models")
parser.add_argument("--start_channel", type=int,
                    dest="start_channel", default=64,  # default:8, 7 for stage
                    help="number of start channels")
parser.add_argument("--datapath", type=str,
                    dest="datapath",
                    default='',
                    help="data path for training images")
parser.add_argument("--freeze_step", type=int,
                    dest="freeze_step", default=3000,
                    help="Number of step to freeze the previous level")
opt = parser.parse_args()

lr = opt.lr
start_channel = opt.start_channel
antifold = opt.antifold
n_checkpoint = opt.checkpoint
datapath = opt.datapath
freeze_step = opt.freeze_step

iteration_lvl1 = opt.iteration_lvl1
iteration_lvl2 = opt.iteration_lvl2
iteration_lvl3 = opt.iteration_lvl3

model_name = "LDR_OASIS_NCC_unit_disp_add_fea64_reg01_10_2D_"


def dice(im1, atlas):
    unique_class = np.unique(atlas)
    dice = 0
    num_count = 0
    for i in unique_class:
        if (i == 0) or ((im1 == i).sum() == 0) or ((atlas == i).sum() == 0):
            continue

        sub_dice = np.sum(atlas[im1 == i] == i) * 2.0 / (np.sum(im1 == i) + np.sum(atlas == i))
        dice += sub_dice
        num_count += 1
        # print(sub_dice)
    # print(num_count, len(unique_class)-1)
    return dice / num_count


def train_lvl1():
    print("Training lvl1...")
    model = CSAIN_disp_lvl1(2, 2, start_channel, is_train=True,
                                                                    imgshape=imgshape_4,
                                                                    range_flow=range_flow).cuda()

    loss_similarity = NCC(win=3)
    loss_smooth = spatially_variant_reg

    transform = SpatialTransform_unit().cuda()

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    # OASIS
    names = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/slice_norm.nii.gz'))
    seg_name = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/slice_seg4.nii.gz'))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model_dir = '../Model/Stage'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((3, iteration_lvl1 + 1))

    training_generator = Data.DataLoader(Dataset_epoch(names, seg_name, norm=True), batch_size=1,
                                         shuffle=True, num_workers=2)
    step = 0
    load_model = False
    if load_model is True:
        model_path = "../Model/LDR_LPBA_NCC_lap_share_preact_1_05_3000.pth"
        print("Loading weight: ", model_path)
        step = 3000
        model.load_state_dict(torch.load(model_path))
        temp_lossall = np.load("../Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy")
        lossall[:, 0:3000] = temp_lossall[:, 0:3000]

    while step <= iteration_lvl1:
        for X, Y, seg_Y in training_generator:

            X = X.squeeze(-1).cuda().float()
            Y = Y.squeeze(-1).cuda().float()
            seg_Y = seg_Y.squeeze(-1).cuda().float()
            reg_code = torch.rand(5, dtype=X.dtype, device=X.device)

            F_X_Y, X_Y, Y_4x, F_xy, _, reg_matrix = model(X, Y, seg_Y, reg_code)
            loss_multiNCC = loss_similarity(X_Y, Y_4x)
            _, _, x, y = F_X_Y.shape
            norm_vector = torch.zeros((1, 2, 1, 1), dtype=F_X_Y.dtype, device=F_X_Y.device)
            norm_vector[0, 0, 0, 0] = (y - 1)
            norm_vector[0, 1, 0, 0] = (x - 1)
            loss_regulation = loss_smooth(F_X_Y * norm_vector, reg_matrix)

            loss = loss_multiNCC + max_smooth * loss_regulation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" -smo "{3:.4f} -reg_c "{4:.4f}"'.format(
                    step, loss.item(), loss_multiNCC.item(), loss_regulation.item(), reg_code[0].item()))
            sys.stdout.flush()

            # with lr 1e-3 + with bias
            if step % n_checkpoint == 0:
                modelname = model_dir + '/' + model_name + "stagelvl1_" + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + '/loss' + model_name + "stagelvl1_" + str(step) + '.npy', lossall)

            step += 1

            if step > iteration_lvl1:
                break
        print("one epoch pass")
    np.save(model_dir + '/loss' + model_name + 'stagelvl1.npy', lossall)


def train_lvl2():
    print("Training lvl2...")
    model_lvl1 = CSAIN_disp_lvl1(2, 2, start_channel, is_train=True,
                                                                         imgshape=imgshape_4,
                                                                         range_flow=range_flow).cuda()

    model_path = sorted(glob.glob("../Model/Stage/" + model_name + "stagelvl1_??????.pth"))[-1]
    model_lvl1.load_state_dict(torch.load(model_path))
    print("Loading weight for model_lvl1...", model_path)

    # Freeze model_lvl1 weight
    for param in model_lvl1.parameters():
        param.requires_grad = False

    model = CSAIN_disp_lvl2(2, 2, start_channel, is_train=True,
                                                                    imgshape=imgshape_2,
                                                                    range_flow=range_flow, model_lvl1=model_lvl1).cuda()

    loss_similarity = multi_resolution_NCC(win=5, scale=2)
    loss_smooth = spatially_variant_reg

    transform = SpatialTransform_unit().cuda()

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    # OASIS
    names = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/slice_norm.nii.gz'))
    seg_name = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/slice_seg4.nii.gz'))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model_dir = '../Model/Stage'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((3, iteration_lvl2 + 1))

    training_generator = Data.DataLoader(Dataset_epoch(names, seg_name, norm=True), batch_size=1,
                                         shuffle=True, num_workers=2)
    step = 0
    load_model = False
    if load_model is True:
        model_path = "../Model/LDR_LPBA_NCC_lap_share_preact_1_05_3000.pth"
        print("Loading weight: ", model_path)
        step = 3000
        model.load_state_dict(torch.load(model_path))
        temp_lossall = np.load("../Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy")
        lossall[:, 0:3000] = temp_lossall[:, 0:3000]

    while step <= iteration_lvl2:
        for X, Y, seg_Y in training_generator:

            X = X.squeeze(-1).cuda().float()
            Y = Y.squeeze(-1).cuda().float()
            seg_Y = seg_Y.squeeze(-1).cuda().float()
            reg_code = torch.rand(5, dtype=X.dtype, device=X.device)

            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, _, reg_matrix = model(X, Y, seg_Y, reg_code)

            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            _, _, x, y = F_X_Y.shape
            norm_vector = torch.zeros((1, 2, 1, 1), dtype=F_X_Y.dtype, device=F_X_Y.device)
            norm_vector[0, 0, 0, 0] = (y - 1)
            norm_vector[0, 1, 0, 0] = (x - 1)
            loss_regulation = loss_smooth(F_X_Y * norm_vector, reg_matrix)

            loss = loss_multiNCC + max_smooth * loss_regulation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" -smo "{3:.4f} -reg_c "{4:.4f}"'.format(
                    step, loss.item(), loss_multiNCC.item(), loss_regulation.item(),
                    reg_code[0].item()))
            sys.stdout.flush()

            # with lr 1e-3 + with bias
            if step % n_checkpoint == 0:
                modelname = model_dir + '/' + model_name + "stagelvl2_" + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + '/loss' + model_name + "stagelvl2_" + str(step) + '.npy', lossall)

            if step == freeze_step:
                model.unfreeze_modellvl1()

            step += 1

            if step > iteration_lvl2:
                break
        print("one epoch pass")
    np.save(model_dir + '/loss' + model_name + 'stagelvl2.npy', lossall)


def train_lvl3():
    print("Training lvl3...")
    model_lvl1 = CSAIN_disp_lvl1(2, 2, start_channel, is_train=True,
                                                                         imgshape=imgshape_4,
                                                                         range_flow=range_flow).cuda()
    model_lvl2 = CSAIN_disp_lvl2(2, 2, start_channel, is_train=True,
                                                                         imgshape=imgshape_2,
                                                                         range_flow=range_flow,
                                                                         model_lvl1=model_lvl1).cuda()

    model_path = sorted(glob.glob("../Model/Stage/" + model_name + "stagelvl2_??????.pth"))[-1]
    model_lvl2.load_state_dict(torch.load(model_path))
    print("Loading weight for model_lvl2...", model_path)

    # Freeze model_lvl1 weight
    for param in model_lvl2.parameters():
        param.requires_grad = False

    model = CSAIN_disp_lvl3(2, 2, start_channel, is_train=True,
                                                                    imgshape=imgshape,
                                                                    range_flow=range_flow, model_lvl2=model_lvl2).cuda()

    loss_similarity = multi_resolution_NCC(win=7, scale=3)
    loss_smooth = spatially_variant_reg

    transform = SpatialTransform_unit().cuda()
    transform_nearest = SpatialTransformNearest_unit().cuda()

    for param in transform.parameters():
        param.requires_grad = False
        param.volatile = True

    # OASIS
    names = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/slice_norm.nii.gz'))
    seg_name = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/slice_seg4.nii.gz'))

    grid_unit = F.affine_grid(torch.eye(3)[0:2].unsqueeze(0), (1,) + (1,) + imgshape, align_corners=True).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model_dir = '../Model'

    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    lossall = np.zeros((3, iteration_lvl3 + 1))

    training_generator = Data.DataLoader(Dataset_epoch(names, seg_name, norm=True), batch_size=1,
                                         shuffle=True, num_workers=2)
    step = 0
    load_model = False
    if load_model is True:
        model_path = "../Model/LDR_LPBA_NCC_lap_share_preact_1_05_3000.pth"
        print("Loading weight: ", model_path)
        step = 3000
        model.load_state_dict(torch.load(model_path))
        temp_lossall = np.load("../Model/loss_LDR_LPBA_NCC_lap_share_preact_1_05_3000.npy")
        lossall[:, 0:3000] = temp_lossall[:, 0:3000]

    while step <= iteration_lvl3:
        for X, Y, seg_Y in training_generator:

            X = X.squeeze(-1).cuda().float()
            Y = Y.squeeze(-1).cuda().float()
            seg_Y = seg_Y.squeeze(-1).cuda().float()
            reg_code = torch.rand(5, dtype=X.dtype, device=X.device)

            F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _, reg_matrix = model(X, Y, seg_Y, reg_code)

            loss_multiNCC = loss_similarity(X_Y, Y_4x)

            _, _, x, y = F_X_Y.shape
            norm_vector = torch.zeros((1, 2, 1, 1), dtype=F_X_Y.dtype, device=F_X_Y.device)
            norm_vector[0, 0, 0, 0] = (y - 1)
            norm_vector[0, 1, 0, 0] = (x - 1)
            loss_regulation = loss_smooth(F_X_Y * norm_vector, reg_matrix)

            loss = loss_multiNCC + max_smooth * loss_regulation

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            lossall[:, step] = np.array(
                [loss.item(), loss_multiNCC.item(), loss_regulation.item()])
            sys.stdout.write(
                "\r" + 'step "{0}" -> training loss "{1:.4f}" - sim_NCC "{2:4f}" -smo "{3:.4f} -reg_c "{4:.4f}"'.format(
                    step, loss.item(), loss_multiNCC.item(), loss_regulation.item(),
                    reg_code[0].item()))
            sys.stdout.flush()

            # with lr 1e-3 + with bias
            if step % n_checkpoint == 0:
                modelname = model_dir + '/' + model_name + "stagelvl3_" + str(step) + '.pth'
                torch.save(model.state_dict(), modelname)
                np.save(model_dir + '/loss' + model_name + "stagelvl3_" + str(step) + '.npy', lossall)

                # Put your validation code here
                # ---------------------------------------
                # OASIS (Validation)
                imgs = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/slice_norm.nii.gz'))[-15:]
                labels = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/slice_seg24.nii.gz'))[-15:]
                seg_name = sorted(glob.glob(datapath + '/OASIS_OAS1_*_MR1/slice_seg4.nii.gz'))[-15:]

                valid_generator = Data.DataLoader(
                    Dataset_epoch_validation(imgs, labels, seg_name, norm=True),
                    batch_size=1,
                    shuffle=False, num_workers=2)

                dice_total = []
                Jacobian_total = []
                grid = F.affine_grid(torch.eye(3)[0:2].unsqueeze(0), (1,) + (1,) + imgshape, align_corners=True).cuda()
                use_cuda = True
                device = torch.device("cuda" if use_cuda else "cpu")
                print("\nValidating...")
                for batch_idx, data in enumerate(valid_generator):
                    X, Y, X_label, Y_label, seg_Y = data[0].squeeze(-1).to(device), data[1].squeeze(-1).to(device), \
                        data[
                            2].squeeze(-1).to(device), data[3].squeeze(-1).to(device), data[
                        4].squeeze(-1).to(device)

                    with torch.no_grad():
                        reg_code = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2], dtype=X.dtype, device=X.device)

                        F_X_Y, X_Y, Y_4x, F_xy, F_xy_lvl1, F_xy_lvl2, _, _ = model(X, Y, seg_Y, reg_code)
                        X_Y_label = transform_nearest(X_label, F_X_Y.permute(0, 2, 3, 1), grid_unit).cpu().numpy()[0, 0,
                                    :, :]
                        Y_label = Y_label.cpu().numpy()[0, 0, :, :]

                        dice_score = dice(np.floor(X_Y_label), np.floor(Y_label))
                        dice_total.append(dice_score)

                        F_X_Y_norm = transform_unit_flow_to_flow_2D(F_X_Y.permute(0, 2, 3, 1).clone())
                        loss_Jacobian = neg_Jdet_loss(F_X_Y_norm, grid).cpu().numpy()
                        Jacobian_total.append(loss_Jacobian)

                print("Dice mean: ", np.mean(dice_total))
                with open(log_dir, "a") as log:
                    log.write(
                        str(step) + ":" + str(np.mean(dice_total)) + "," + str(np.mean(Jacobian_total)) + "\n")

            if step == freeze_step:
                model.unfreeze_modellvl2()

            step += 1

            if step > iteration_lvl3:
                break
        print("one epoch pass")
    np.save(model_dir + '/loss' + model_name + 'stagelvl3.npy', lossall)


if __name__ == "__main__":
    imgshape = (160, 192)
    imgshape_4 = (160 // 4, 192 // 4)
    imgshape_2 = (160 // 2, 192 // 2)

    # Create and initalize log file
    if not os.path.isdir("../Log"):
        os.mkdir("../Log")

    log_dir = "../Log/" + model_name + ".txt"

    with open(log_dir, "a") as log:
        log.write("Validation Dice log for " + model_name[0:-1] + ":\n")

    range_flow = 0.4
    max_smooth = 10.
    start_t = datetime.now()
    train_lvl1()
    train_lvl2()
    train_lvl3()
    # time
    end_t = datetime.now()
    total_t = end_t - start_t
    print("Time: ", total_t.total_seconds())
