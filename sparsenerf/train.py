import os

import imageio
import numpy as np
import torch

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm

from sparsenerf.networks import LearnFocal, LearnPose, TinyNerf, UNet, inv_c2w
from sparsenerf.nerfmm.utils.comp_ray_dir import comp_ray_dir_cam_fxfy
from sparsenerf.nerfmm.utils.volume_op import volume_rendering, volume_sampling_ndc
from sparsenerf.nerfmm.utils.pose_utils import create_spiral_poses

from sparsenerf.nerfmm.utils.pos_enc import encode_position
from sparsenerf.nerfmm.utils.training_utils import mse2psnr

class SparseNeRFScene:
    def __init__(self,
        im_encoder_net,
        im_encoder_out_features,
        N_EPOCH,
        EVAL_INTERVAL,
        ray_params,
        train_imgs,
        scene_name=None,
        RUN_NAME=None,
        learning_rate = 0.001,
        use_im_encoder=False,
        use_views=False,
        train_im_encoder=False,
        n_views=-1,
        mask_current_view=True,
        disable_writer=False,
        device='cuda'
    ):
        self.device = device

        self.H = train_imgs.shape[1]
        self.W = train_imgs.shape[2]

        # Initialise all trainabled parameters
        self.focal_net = LearnFocal(self.H, self.W, req_grad=True).to(self.device)
        self.pose_param_net = LearnPose(num_cams=train_imgs.shape[0], learn_R=True, learn_t=True).to(self.device)

        if n_views == -1:
            self.n_views = train_imgs.shape[0]
        else:
            self.n_views = n_views

        img_features = None
        if use_views:
            img_features = 3 * self.n_views
            if use_im_encoder:
                img_features = im_encoder_out_features * self.n_views

        # Get a tiny NeRF model. Hidden dimension set to 128
        self.nerf_model = TinyNerf(pos_in_dims=63, dir_in_dims=27, D=128, image_feat_dim=img_features).to(self.device)

        self.im_encoder_net = im_encoder_net
        self.N_EPOCH = N_EPOCH
        self.EVAL_INTERVAL = EVAL_INTERVAL
        self.ray_params = ray_params
        self.train_imgs = train_imgs
        self.scene_name = scene_name
        self.RUN_NAME = RUN_NAME
        self.use_im_encoder = use_im_encoder
        self.use_views = use_views
        self.train_im_encoder = train_im_encoder
        self.mask_current_view = mask_current_view

        # Set lr and scheduler: these are just stair-case exponantial decay lr schedulers.
        opt_nerf = torch.optim.Adam(self.nerf_model.parameters(), lr=learning_rate)
        opt_focal = torch.optim.Adam(self.focal_net.parameters(), lr=learning_rate)
        opt_pose = torch.optim.Adam(self.pose_param_net.parameters(), lr=learning_rate)
        self.opt_enc = torch.optim.Adam(self.im_encoder_net.parameters(), lr=learning_rate/10)
        self.optimisers = [opt_nerf, opt_focal, opt_pose]

        scheduler_nerf = MultiStepLR(opt_nerf, milestones=list(range(0, 10000*10, 10*10)), gamma=0.9954)
        scheduler_focal = MultiStepLR(opt_focal, milestones=list(range(0, 10000*10, 100*10)), gamma=0.9)
        scheduler_pose = MultiStepLR(opt_pose, milestones=list(range(0, 10000*10, 100*10)), gamma=0.9)
        scheduler_enc = MultiStepLR(self.opt_enc, milestones=list(range(0, 10000, 100*10)), gamma=0.9)
        self.schedulers = [scheduler_nerf, scheduler_focal, scheduler_pose]
        if train_im_encoder:
            self.schedulers.append(scheduler_enc)

        # training stuff
        self.epoch = 0
        # Set tensorboard writer
        if self.scene_name is not None and self.RUN_NAME is not None:
            log_path = os.path.join('logs', self.scene_name, self.RUN_NAME)
            if os.path.exists(log_path):
                raise Exception(f"Run {self.RUN_NAME} already exists")

            self.writer = SummaryWriter(log_dir=log_path)
            self.writer.add_hparams({
                'num_images': self.train_imgs.shape[0],
                'positional_encoding_reg_percent': self.ray_params.POS_ENC_REG_ITERS_p,
                'directional_encoding_reg_percent': self.ray_params.DIR_ENC_REG_ITERS_p,
                'use_im_encoder': use_im_encoder,
                'use_views': use_views,
                'train_im_encoder': train_im_encoder,
                'n_views': n_views,
                'mask_current_view': mask_current_view,
            }, {})

        # placeholder views
        if use_views:
            c2ws = [self.pose_param_net(i) for i in range(n_views)]
            self.learned_c2ws = torch.stack(c2ws)
            self.learned_c2ws_inv = torch.stack([inv_c2w(c2w) for c2w in c2ws])
    
    def _backproject_world_poses(self, sample_pos, fxfy, H, W, current_image=None):
        # rotation + translation
        R = self.learned_c2ws_inv[:self.n_views, :3, :3] # (N_cam, 3, 3)
        t = self.learned_c2ws_inv[:self.n_views, :3, 3]  # (N_cam, 3)

        H_rays, W_rays, N_samples, _ = sample_pos.shape
        N_cam = self.n_views

        # expand points for broadcasting
        pts = sample_pos.unsqueeze(0)                    # (1, 32, 32, N_sample, 3)

        # world -> camera
        pts_cam = torch.einsum(
            'cij,cwhnj->cwhni',
            R,
            pts.expand(R.shape[0], -1, -1, -1, -1)
        ) + t[:, None, None, None, :] # (N_cam, W, H, N_sample, 3)

        uv, _ = pts_cam_to_img(pts_cam, fxfy[0], fxfy[1], H, W) # (N_cam, 32, 32, N_sample, C)
        uv = uv.detach()

        # pixel coordinates
        u = uv[..., 0]   # (N_cam, 32, 32, N_sample)
        v = uv[..., 1]   # (N_cam, 32, 32, N_sample)

        grid_y = 2.0 * (u / (H - 1)) - 1.0
        grid_x = 2.0 * (v / (W - 1)) - 1.0

        grid = torch.stack([grid_x, grid_y], dim=-1) # (N_cam, H_rays, W_rays, N_samples, 2)
        grid = grid.reshape(N_cam, H_rays, W_rays * N_samples, 2)

        sampled = F.grid_sample(
            self.encoded_ims.permute(0, 3, 1, 2),
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        ) # (N_cam, C, H_rays, W_rays * N_samples)

        sampled = sampled.reshape(N_cam, sampled.shape[1], H_rays, W_rays, N_samples)
        sampled = sampled.permute(2, 3, 4, 0, 1)  # (H_rays, W_rays, N_samples, N_cam, C)

        if self.mask_current_view and current_image is not None and current_image < N_cam:
            sampled[:, :, :, current_image, :] = 0.0

        return sampled.reshape(H_rays, W_rays, N_samples, -1)  # (H_rays, W_rays, N_samples, N_cam * C)


    def _model_render_image(self, H, W, c2w, rays_cam, t_vals, fxfy, perturb_t, sigma_noise_std, image_i=None):
        """
        :param c2w:         (4, 4)                  pose to transform ray direction from cam to world.
        :param rays_cam:    (someH, someW, 3)       ray directions in camera coordinate, can be random selected
                                                    rows and cols, or some full rows, or an entire image.
        :param t_vals:      (N_samples)             sample depth along a ray.
        :param perturb_t:   True/False              perturb t values.
        :param sigma_noise_std: float               add noise to raw density predictions (sigma).
        :return:            (someH, someW, 3)       volume rendered images for the input rays.
        """
        # KEY 2: sample the 3D volume using estimated poses and intrinsics online.
        # (H, W, N_sample, 3), (H, W, 3), (H, W, N_sam)
        sample_pos, _, ray_dir_world, t_vals_noisy = volume_sampling_ndc(c2w, rays_cam, t_vals, self.ray_params.NEAR,
                                                                        self.ray_params.FAR, H, W, fxfy, perturb_t)

        # encode position: (H, W, N_sample, (2L+1)*C = 63)
        pos_enc = encode_position_regularized(sample_pos, levels=self.ray_params.POS_ENC_FREQ, inc_input=True, current_iter=self.epoch, total_reg_iter=self.ray_params.POS_ENC_REG_ITERS)

        # encode direction: (H, W, N_sample, (2L+1)*C = 27)
        ray_dir_world = F.normalize(ray_dir_world, p=2, dim=2)  # (H, W, 3)
        dir_enc = encode_position_regularized(ray_dir_world, levels=self.ray_params.DIR_ENC_FREQ, inc_input=True, current_iter=self.epoch, total_reg_iter=self.ray_params.DIR_ENC_REG_ITERS)  # (H, W, 27)
        dir_enc = dir_enc.unsqueeze(2).expand(-1, -1, self.ray_params.N_SAMPLE, -1)  # (H, W, N_sample, 27)

        img_feat = None
        if self.use_views:
            img_feat = self._backproject_world_poses(sample_pos, fxfy, H, W, image_i)

        # inference rgb and density using position and direction encoding.
        rgb_density = self.nerf_model(pos_enc, dir_enc, img_feat)  # (H, W, N_sample, 4)

        render_result = volume_rendering(rgb_density, t_vals_noisy, sigma_noise_std, rgb_act_fn=torch.sigmoid)
        rgb_rendered = render_result['rgb']  # (H, W, 3)
        depth_map = render_result['depth_map']  # (H, W)

        result = {
            'rgb': rgb_rendered,  # (H, W, 3)
            'depth_map': depth_map,  # (H, W)
        }

        return result


    def _train_one_epoch_inner(self):
        self.nerf_model.train()
        self.focal_net.train()
        self.pose_param_net.train()
        if self.train_im_encoder:
            self.im_encoder_net.train()

        t_vals = torch.linspace(self.ray_params.NEAR, self.ray_params.FAR, self.ray_params.N_SAMPLE, device=self.device)  # (N_sample,) sample position
        L2_loss_epoch = []

        # shuffle the training imgs
        ids = np.arange(self.train_imgs.shape[0])
        np.random.shuffle(ids)

        for i in ids:
            if self.use_views:
                if self.use_im_encoder:
                    inputs = self.train_imgs[:self.n_views].permute(0, 3, 1, 2).to(self.device)
                    self.encoded_ims = self.im_encoder_net(inputs).permute(0, 2, 3, 1)
                else:
                    self.encoded_ims = self.train_imgs[:self.n_views].to(self.device) # (N, H, W, C)

                self.im_features_downsampling_ratio = torch.tensor([
                    self.train_imgs.shape[1]/self.encoded_ims.shape[1],
                    self.train_imgs.shape[2]/self.encoded_ims.shape[2]
                ], device=self.encoded_ims.device)

            fxfy = self.focal_net()

            # KEY 1: compute ray directions using estimated intrinsics online.
            ray_dir_cam = comp_ray_dir_cam_fxfy(self.H, self.W, fxfy[0], fxfy[1])
            img = self.train_imgs[i].to(self.device)  # (H, W, 4)
            c2w = self.pose_param_net(i)  # (4, 4)

            # sample 32x32 pixel on an image and their rays for training.
            r_id = torch.randperm(self.H, device=self.device)[:32]  # (N_select_rows)
            c_id = torch.randperm(self.W, device=self.device)[:32]  # (N_select_cols)
            ray_selected_cam = ray_dir_cam[r_id][:, c_id]  # (N_select_rows, N_select_cols, 3)
            img_selected = img[r_id][:, c_id]  # (N_select_rows, N_select_cols, 3)

            # render an image using selected rays, pose, sample intervals, and the network
            render_result = self._model_render_image(self.H, self.W, c2w, ray_selected_cam, t_vals, fxfy, perturb_t=True, sigma_noise_std=0.0, image_i=i)
            rgb_rendered = render_result['rgb']  # (N_select_rows, N_select_cols, 3)
            L2_loss = F.mse_loss(rgb_rendered, img_selected)  # loss for one image

            # If we are training the im encoder, we have to keep the graph between runs.
            # On the last run we can free it
            L2_loss.backward()

            for opt in self.optimisers: opt.step()
            for opt in self.optimisers: opt.zero_grad()
            if self.train_im_encoder:
                self.opt_enc.step()
                self.opt_enc.zero_grad()

            L2_loss_epoch.append(L2_loss)
        

        L2_loss_epoch_mean = torch.stack(L2_loss_epoch).mean().item()
        return L2_loss_epoch_mean

    def render_novel_view(self, H, W, c2w, fxfy):
        self.nerf_model.eval()

        ray_dir_cam = comp_ray_dir_cam_fxfy(H, W, fxfy[0], fxfy[1])
        t_vals = torch.linspace(self.ray_params.NEAR, self.ray_params.FAR, self.ray_params.N_SAMPLE, device=self.device)  # (N_sample,) sample position

        c2w = c2w.to(self.device)  # (4, 4)

        # split an image to rows when the input image resolution is high
        rays_dir_cam_split_rows = ray_dir_cam.split(10, dim=0)  # input 10 rows each time
        rendered_img = []
        rendered_depth = []
        for rays_dir_rows in rays_dir_cam_split_rows:
            render_result = self._model_render_image(H, W, c2w, rays_dir_rows, t_vals, fxfy, perturb_t=False, sigma_noise_std=0.0)
            rgb_rendered_rows = render_result['rgb']  # (num_rows_eval_img, W, 3)
            depth_map = render_result['depth_map']  # (num_rows_eval_img, W)

            rendered_img.append(rgb_rendered_rows)
            rendered_depth.append(depth_map)

        # combine rows to an image
        rendered_img = torch.cat(rendered_img, dim=0)  # (H, W, 3)
        rendered_depth = torch.cat(rendered_depth, dim=0)  # (H, W)
        return rendered_img, rendered_depth

    def train_epoch(self):
        L2_loss = self._train_one_epoch_inner()
        train_psnr = mse2psnr(L2_loss)
        if self.writer is not None:
            self.writer.add_scalar('train/psnr', train_psnr, self.epoch)
        
        fxfy = self.focal_net()
        print('epoch {0:4d} Training PSNR {1:.3f}, estimated fx {2:.1f} fy {3:.1f}'.format(self.epoch, train_psnr, fxfy[0], fxfy[1]))

        for sc in self.schedulers: sc.step()

        if self.use_views:
            c2ws = [self.pose_param_net(i) for i in range(self.n_views)]
            self.learned_c2ws = torch.stack(c2ws)
            self.learned_c2ws_inv = torch.stack([inv_c2w(c2w) for c2w in c2ws])

        if self.writer is not None:
            with torch.no_grad():
                if (self.epoch+1) % self.EVAL_INTERVAL == 0:
                    eval_c2w = torch.eye(4, dtype=torch.float32)  # (4, 4)
                    fxfy = self.focal_net()
                    rendered_img, rendered_depth = self.render_novel_view(self.H, self.W, eval_c2w, fxfy)
                    self.writer.add_image('eval/img', rendered_img.permute(2, 0, 1), global_step=self.epoch)
                    self.writer.add_image('eval/depth', rendered_depth.unsqueeze(0), global_step=self.epoch)
        
        self.epoch += 1
    
    def generate_results(self, resize_ratio = 2):
        # Render novel views from a sprial camera trajectory.
        # The spiral trajectory generation function is modified from https://github.com/kwea123/nerf_pl.

        # Render full images are time consuming, especially on colab so we render a smaller version instead.
        with torch.no_grad():
            optimised_poses = torch.stack([self.pose_param_net(i) for i in range(self.train_imgs.shape[0])])
            radii = np.percentile(np.abs(optimised_poses.cpu().numpy()[:, :3, 3]), q=50, axis=0)  # (3,)
            spiral_c2ws = create_spiral_poses(radii, focus_depth=3.5, n_poses=30, n_circle=1)
            spiral_c2ws = torch.from_numpy(spiral_c2ws).float()  # (N, 3, 4)

            # change intrinsics according to resize ratio
            fxfy = self.focal_net()
            novel_fxfy = fxfy / resize_ratio
            novel_H, novel_W = self.H // resize_ratio, self.W // resize_ratio

            print('NeRF trained in {0:d} x {1:d} for {2:d} epochs'.format(self.H, self.W, self.N_EPOCH))
            print('Rendering novel views in {0:d} x {1:d}'.format(novel_H, novel_W))

            novel_img_list, novel_depth_list = [], []
            for i in tqdm(range(spiral_c2ws.shape[0]), desc='novel view rendering'):
                novel_img, novel_depth = self.render_novel_view(novel_H, novel_W, spiral_c2ws[i], novel_fxfy)
                novel_img_list.append(novel_img)
                novel_depth_list.append(novel_depth)

            print('Novel view rendering done. Saving to GIF images...')
            novel_img_list = (torch.stack(novel_img_list) * 255).cpu().numpy().astype(np.uint8)
            novel_depth_list = (torch.stack(novel_depth_list) * 200).cpu().numpy().astype(np.uint8)  # depth is always in 0 to 1 in NDC

            if self.scene_name is not None and self.RUN_NAME is not None:
                os.makedirs(os.path.join('nvs_results', self.RUN_NAME), exist_ok=True)
                imageio.mimwrite(os.path.join('nvs_results', self.RUN_NAME, self.scene_name + '_img.gif'), novel_img_list, fps=30, loop=0)
                imageio.mimwrite(os.path.join('nvs_results', self.RUN_NAME, self.scene_name + '_depth.gif'), novel_depth_list, fps=30, loop=0)
                print('GIF images saved.')

memo = None
def get_freq_reg_mask(pos_enc_length, current_iter, total_reg_iter, device):
    """
    Returns a frequency mask for position encoding
    """
    global memo
    if memo is not None and (pos_enc_length, current_iter, total_reg_iter) == memo[0]:
        return memo[1].detach().clone()

    if current_iter < total_reg_iter:
        freq_mask = torch.zeros(pos_enc_length, device=device)

        ptr = pos_enc_length / 3 * current_iter / total_reg_iter + 1
        ptr = min(ptr, pos_enc_length / 3)

        int_ptr = int(ptr)

        # integer part
        if int_ptr > 0:
            freq_mask[: int_ptr * 3] = 1.0

        # fractional part
        frac_start = int_ptr * 3
        frac_end = frac_start + 3
        if frac_start < pos_enc_length:
            frac_val = ptr - int_ptr
            freq_mask[frac_start: min(frac_end, pos_enc_length)] = frac_val

        # numerical stability
        freq_mask = torch.clamp(freq_mask, 1e-8, 1 - 1e-8)

        out = freq_mask

    else:
        out = torch.ones(pos_enc_length, device=device)

    memo = ((pos_enc_length, current_iter, total_reg_iter), out)
    return out

def encode_position_regularized(input, levels, inc_input, current_iter, total_reg_iter):
    raw = encode_position(input, levels, inc_input)
    
    if current_iter is None:
        return raw
    
    mask = get_freq_reg_mask(raw.shape[-1], current_iter, total_reg_iter, input.device)
    return raw * mask

def pts_cam_to_img(pts_cam, fx, fy, H, W):
    """
    Project camera-space points into image pixel coordinates.

    Args:
        pts_cam: (..., 3)
                 camera-space points

        fx, fy: focal lengths

        H, W: image height/width

    Returns:
        uv: (..., 2)
            pixel coordinates (x, y)

        depth: (...)
               positive forward depth
    """
    x = pts_cam[..., 0]
    y = pts_cam[..., 1]
    z = pts_cam[..., 2]

    # OpenGL convention:
    # forward = -z
    depth = -z

    u = -fy * (y / depth) + 0.5 * H
    v = fx * (x / depth) + 0.5 * W

    uv = torch.stack([u, v], dim=-1)

    return uv, depth