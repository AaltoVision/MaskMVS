from __future__ import division
import torch
import numpy as np
from torch import nn
from torch.autograd import Variable



def generate_volume(tgt_img, n_img, intrinsics, intrinsics_inv, pose, D):
    b, _, h, w = tgt_img.size()
    volume = [tgt_img]

    for k, d in enumerate(D):
        warped_img = warping_neighbor(n_img, pose, d, intrinsics, intrinsics_inv)
        volume.append(warped_img)

    return torch.cat(volume, dim=1).cuda()



def warping_neighbor(img, pose_var, d, intrinsics, intrinsics_inv):
    b, _, h, w = img.size()

    indy = Variable(torch.arange(0, h).view(1, h, 1).expand(1, h, w).float().cuda())
    indx = Variable(torch.arange(0, w).view(1, 1, w).expand(1, h, w).float().cuda())

    ones = Variable(torch.ones(1, h, w).float().cuda())
    pixel_coords = torch.stack((indx, indy, ones), dim=1).expand(b, 3, h, w)  # [1, 3, H, W]

    R = pose_var[:, :3, :3]

    t = pose_var[:, :3, 3].contiguous().view(b, 3, 1)


    n = torch.from_numpy(np.matrix([0, 0, -1]))
    n = Variable(n.expand(b, 1, 3).float().cuda())


    H1 = intrinsics.bmm((R - (t.bmm(n)) / d).bmm(intrinsics_inv))


    map_coords = (H1.bmm(pixel_coords.contiguous().view(b, 3, -1))).view(b, 3, h, w)

    X = map_coords[:, 0]
    Y = map_coords[:, 1]
    Z = map_coords[:, 2].clamp(min=1e-3)

    X_norm = 2 * (X / Z) / (w - 1) - 1
    Y_norm = 2 * (Y / Z) / (h - 1) - 1

    X_mask = ((X_norm > 1) + (X_norm < -1)).detach()
    X_norm[X_mask] = 2 
    Y_mask = ((Y_norm > 1) + (Y_norm < -1)).detach()
    Y_norm[Y_mask] = 2

    map_coords = (torch.stack([X_norm, Y_norm], dim=3)).view(b, h, w, 2)
    projected_img = torch.nn.functional.grid_sample(img, map_coords, padding_mode='zeros')

    return projected_img

def where(cond, x_1, x_2):
    return (cond * x_1) + ((1-cond) * x_2)

def gen_mask_gt(gt, D):
    #return B,D,H,W
    mask = []
    valid = gt!=0 
    for d in D:     
        mask.append(where((1/gt <= d).type(torch.FloatTensor), 1, 0))
    return valid, torch.stack(mask, dim = 1).cuda() #B,D,H,W



