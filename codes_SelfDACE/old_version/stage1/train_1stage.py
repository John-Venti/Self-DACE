import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import Myloss
import numpy as np
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter



def reflection(im):
    mr, mg, mb = torch.split(im, 1, dim=1)
    r = mr / (mr + mg + mb + 0.0001)
    g = mg / (mr + mg + mb + 0.0001)
    b = mb / (mr + mg + mb + 0.0001)
    return torch.cat([r, g, b], dim=1)


def luminance(s):
    return ((s[:, 0, :, :] + s[:, 1, :, :] + s[:, 2, :, :])).unsqueeze(1)

def train(config):
    os.environ['CUDA_VISIBLE_DEVICES']='0'

    light_net = model.light_net().cuda()
    optimizer = torch.optim.Adam(light_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if config.load_pretrain == True:
        light_net.load_state_dict(torch.load(config.pretrain_dir)['net'])
        optimizer.load_state_dict(torch.load(config.pretrain_dir)['optimizer'])
    train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

    L_gcolor = Myloss.L_color()
    L_exp = Myloss.L_exp(4, 0.8)
    L_smo = Myloss.L_SMO()
    L_locl = Myloss.L_localcolor()

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.99)

    light_net.train()
    loss_idx_value = 0
    for epoch in range(config.num_epochs):
        for iteration, img_lowlight in enumerate(train_loader):
            img_lowlight = img_lowlight.cuda()
            # n, c, h, w = img_lowlight.size()
            # sigma = 0.005 * np.random.rand(1)
            # # sigma = 0.01
            # gaussian_noise = np.zeros((3, h, w), dtype=np.float32)
            # noise_r = np.random.normal(0.0, sigma, (1, h, w)).astype(np.float32)
            # noise_g = np.random.normal(0.0, sigma/2, (1, h, w)).astype(np.float32)
            # noise_b = np.random.normal(0.0, sigma, (1, h, w)).astype(np.float32)
            # gaussian_noise += np.concatenate((noise_r, noise_g, noise_b), axis=0)
            # gaussian_noise = torch.from_numpy(gaussian_noise)
            # gaussian_noise = gaussian_noise.repeat([n, 1, 1, 1])
            #
            # # img_re = reflection(img_lowlight)
            # img_lu = luminance(img_lowlight)
            # #
            # img_lowlight_noise = (img_lowlight + (1 - img_lu / 3) * gaussian_noise.cuda()).clip(max=1, min=0)
            #
            # img_re_no = reflection(img_lowlight_noise)
            # img_lu_no = luminance(img_lowlight_noise)

            enhanced_image,  rr1, rr2  = light_net(img_lowlight)
            loss_smo = 1000 * L_smo(rr1) + 5000 * L_smo(rr2)  # 1000
            loss_exp = 5 * torch.mean(L_exp(enhanced_image, img_lowlight))  # 5
            loss_locl = 1000 * torch.mean(L_locl(enhanced_image, img_lowlight))  # 800
            loss_gcol = 1500 * torch.mean(L_gcolor(enhanced_image))  # 20

            loss = loss_exp + loss_smo + loss_locl + loss_gcol# + loss_spa#+ loss_col# + Loss_TV#+ loss_spa# + loss_smooth #+ loss_noise + loss_exp+ loss_col+ loss_spa  Loss_TV +  loss_locl
            loss_idx_value += 1

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(light_net.parameters(),config.grad_clip_norm)
            optimizer.step()


            if ((iteration + 1) % config.display_iter) == 0:
                print("Loss at iteration", iteration + 1, ":", loss.item())
            if ((iteration + 1) % config.snapshot_iter) == 0:
                torch.save({'net': light_net.state_dict(), 'optimizer': optimizer.state_dict()},
                           config.snapshots_folder + "Epoch" + str(epoch) + '.pth')
            scheduler.step()



if __name__ == "__main__":
    start1 = time.perf_counter()
    parser = argparse.ArgumentParser()

    # Input Parameters
    parser.add_argument('--lowlight_images_path', type=str, default="data/train_data/")
    parser.add_argument('--lr', type=float, default=0.00001)#0.00001

    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)

    parser.add_argument('--num_epochs', type=int, default=500)  # initial value 200
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots_light/")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--pretrain_dir', type=str, default= "snapshots/pretrained.pth")

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)


    train(config)
    end1 = time.perf_counter()
    print("final is in : %s Seconds " % (end1 - start1))








