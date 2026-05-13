import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from sparsenerf.nerfmm.utils.lie_group_helper import convert3x4_4x4

def vec2skew(v):
    """
    :param v:  (3, ) torch tensor
    :return:   (3, 3)
    """
    zero = torch.zeros(1, dtype=torch.float32, device=v.device)
    skew_v0 = torch.cat([ zero,    -v[2:3],   v[1:2]])  # (3, 1)
    skew_v1 = torch.cat([ v[2:3],   zero,    -v[0:1]])
    skew_v2 = torch.cat([-v[1:2],   v[0:1],   zero])
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=0)  # (3, 3)
    return skew_v  # (3, 3)


def Exp(r):
    """so(3) vector to SO(3) matrix
    :param r: (3, ) axis-angle, torch tensor
    :return:  (3, 3)
    """
    skew_r = vec2skew(r)  # (3, 3)
    norm_r = r.norm() + 1e-15
    eye = torch.eye(3, dtype=torch.float32, device=r.device)
    R = eye + (torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)
    return R

def inv_c2w(c2w):
    """
    Invert a 4x4 camera-to-world pose matrix.

    Args:
        c2w: (..., 4, 4)

    Returns:
        w2c: (..., 4, 4)
    """
    R = c2w[..., :3, :3]
    t = c2w[..., :3, 3]

    R_inv = R.transpose(-1, -2)
    t_inv = -(R_inv @ t.unsqueeze(-1)).squeeze(-1)

    w2c = torch.eye(
        4,
        dtype=c2w.dtype,
        device=c2w.device
    ).expand(*c2w.shape[:-2], 4, 4).clone()

    w2c[..., :3, :3] = R_inv
    w2c[..., :3, 3] = t_inv

    return w2c

def make_c2w(r, t):
    """
    :param r:  (3, ) axis-angle             torch tensor
    :param t:  (3, ) translation vector     torch tensor
    :return:   (4, 4)
    """
    R = Exp(r)  # (3, 3)
    c2w = torch.cat([R, t.unsqueeze(1)], dim=1)  # (3, 4)
    c2w = convert3x4_4x4(c2w)  # (4, 4)
    return c2w

class LearnFocal(nn.Module):
    def __init__(self, H, W, req_grad):
        super(LearnFocal, self).__init__()
        self.H = H
        self.W = W
        self.fx = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )
        self.fy = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )

    def forward(self):
        # order = 2, check our supplementary.
        fxfy = torch.stack([self.fx**2 * self.W, self.fy**2 * self.H])
        return fxfy

class LearnPose(nn.Module):
    def __init__(self, num_cams, learn_R, learn_t):
        super(LearnPose, self).__init__()
        self.num_cams = num_cams
        self.r = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_R)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(size=(num_cams, 3), dtype=torch.float32), requires_grad=learn_t)  # (N, 3)

    def forward(self, cam_id):
        r = self.r[cam_id]  # (3, ) axis-angle
        t = self.t[cam_id]  # (3, )
        c2w = make_c2w(r, t)  # (4, 4)
        return c2w

class TinyNerf(nn.Module):
    def __init__(self, pos_in_dims, dir_in_dims, D, image_feat_dim = None):
        """
        :param pos_in_dims: scalar, number of channels of encoded positions
        :param dir_in_dims: scalar, number of channels of encoded directions
        :param D:           scalar, number of hidden dimensions
        """
        super(TinyNerf, self).__init__()

        self.pos_in_dims = pos_in_dims
        self.dir_in_dims = dir_in_dims
        self.image_feat_dim = image_feat_dim

        self.layers0 = nn.Sequential(
            nn.Linear(pos_in_dims + (image_feat_dim or 0), D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
            nn.Linear(D, D), nn.ReLU(),
        )

        self.fc_density = nn.Linear(D, 1)
        self.fc_feature = nn.Linear(D, D)
        self.rgb_layers = nn.Sequential(nn.Linear(D + dir_in_dims, D//2), nn.ReLU())
        self.fc_rgb = nn.Linear(D//2, 3)

        self.fc_density.bias.data = torch.tensor([0.1]).float()
        self.fc_rgb.bias.data = torch.tensor([0.02, 0.02, 0.02]).float()

    def forward(self, pos_enc, dir_enc, image_feat = None):
        """
        :param pos_enc: (H, W, N_sample, pos_in_dims) encoded positions
        :param dir_enc: (H, W, N_sample, dir_in_dims) encoded directions
        :param image_feat: (H, W, N_sample, image_feat_dim) image features or None
        :return: rgb_density (H, W, N_sample, 4)
        """
        if image_feat is not None and self.image_feat_dim is None:
            raise Exception("Please set image_feat_dim to use image features")

        if image_feat is not None:
            x = torch.cat([pos_enc, image_feat], dim=3) # (H, W, N_sample, D+image_feat_dim)
        else:
            x = pos_enc

        x = self.layers0(x)  # (H, W, N_sample, D)
        density = self.fc_density(x)  # (H, W, N_sample, 1)

        feat = self.fc_feature(x)  # (H, W, N_sample, D)
        x = torch.cat([feat, dir_enc], dim=3)  # (H, W, N_sample, D+dir_in_dims)
        x = self.rgb_layers(x)  # (H, W, N_sample, D/2)
        rgb = self.fc_rgb(x)  # (H, W, N_sample, 3)

        rgb_den = torch.cat([rgb, density], dim=3)  # (H, W, N_sample, 4)
        return rgb_den

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels = 4, depth=3, first_channels=16, use_skip=True):
        super(UNet, self).__init__()

        self.double_conv_left= nn.ModuleList()
        self.double_conv_right = nn.ModuleList()
        self.conv_up = nn.ModuleList()
        self.pool_down = nn.ModuleList()
        self.depth = depth
        self.use_skip = use_skip
        self.out_channels = out_channels

        self.relu = nn.ReLU()

        prev_channels = in_channels
        channels = first_channels//2

        for l in range(depth):
            channels *= 2
            self.double_conv_left.append(nn.Sequential(
                nn.Conv2d(prev_channels, channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.ReLU(),
            ))

            if l != depth-1:
                self.pool_down.append(nn.MaxPool2d(2,2))
                self.double_conv_right.append(nn.Sequential(
                    nn.Conv2d(channels*2 if self.use_skip else channels, channels, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.ReLU(),
                ))
            
            if l != 0:
                self.conv_up.append(nn.ConvTranspose2d(channels, channels//2 if self.use_skip else channels, kernel_size=2, stride=2))

            prev_channels = channels
        
        self.last_conv = nn.Conv2d(first_channels, out_channels, kernel_size=1)

        self.apply(initialize_weights)

    def forward(self, x : Tensor):
        x = F.interpolate(
            x,
            size=(512, 512),
            mode='bilinear',   # common for images
            align_corners=False
        )
        skip_data : list[Tensor] = []
        for l in range(self.depth):
            x = self.double_conv_left[l](x)

            if l != self.depth-1:
                skip_data.append(x)
                x = self.pool_down[l](x)
        
        for l in range(self.depth-1, 0, -1):
            x = self.conv_up[l-1](x)

            if self.use_skip:
                x = torch.concatenate([x, skip_data[l-1]], dim=1)
            
            x = self.double_conv_right[l-1](x)

        return self.last_conv(x)