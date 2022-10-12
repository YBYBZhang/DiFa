import os
import math
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

import numpy as np

from ZSSGAN.criteria.clip_loss import DirectionLoss
from ZSSGAN.model.psp import pSp


def adjust_sigmoid(x, beta=1):
    return torch.sigmoid(beta * x)


class PSPLoss(torch.nn.Module):
    def __init__(self, device, args=None):
        super(PSPLoss, self).__init__()

        self.device = device
        self.args = args
        self.n_latent = int(math.log(args.size, 2)) * 2 - 2

        self.model = pSp(self.args.psp_path, device, output_size=args.size, has_decoder=False)
        self.model.to(device)

        # Moving Average Coefficient
        self.beta = 0.02
        self.source_mean = self.get_source_mean()
        self.target_mean = self.source_mean
        self.target_set = []
        self.source_set = []
        self.source_pos = 0
        self.target_pos = 0

        resize = (192, 256) if self.args.dataset == 'car' else (256, 256)
        self.psp_preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0]),  # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].]
                                    transforms.Resize(resize),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        self.target_direction = self.get_target_direction()
        self.direction_loss = DirectionLoss('cosine')
        self.iter_diff = []
        self.iter_mean = []
        self.iter_sim = []

    def get_source_mean(self):
        source_path = f"../weights/{self.model.psp_encoder}_source/{self.args.dataset}_A_mean_w.npy"
        if os.path.exists(source_path):
            source_codes = np.load(source_path).reshape((1, self.n_latent, 512))
        else:
            source_codes = np.zeros((1, self.n_latent, 512))
            print(f"There is NO file named {source_path} !")

        unmasked_num = self.n_latent
        if self.args.num_keep_first > 0:
            unmasked_num = self.args.num_keep_first
            unmasked_num = max(unmasked_num, 1)
            source_codes = source_codes.reshape((self.n_latent, 512))[0:unmasked_num]
        return source_codes

    def get_target_direction(self, normalize=True):
        # delta_w_path = os.path.join(self.args.output_dir, 'w_delta.npy')
        delta_w_path = os.path.join(self.args.output_dir, f"{self.args.delta_w_type}_w.npy")

        if os.path.exists(delta_w_path):
            delta_w = np.load(delta_w_path)
        else:
            delta_w = np.ones((self.n_latent, 512))
        unmasked_num = self.n_latent
        if self.args.num_keep_first > 0:
            unmasked_num = self.args.num_keep_first
            unmasked_num = max(unmasked_num, 1)
            delta_w = delta_w[0: unmasked_num]
        
        delta_w = torch.from_numpy(delta_w).to(self.device).float().flatten()
        num_channel = len(delta_w)
        order = delta_w.abs().argsort()
        chosen_order = order[0:int(self.args.psp_alpha * num_channel)]
        # chosen_order = order[-int(self.args.psp_alpha * num_channel)::]  # Choose most important channels
        self.cond = torch.zeros(num_channel).to(self.device)
        self.cond[chosen_order] = 1
        self.cond = self.cond.unsqueeze(0)

        print(f"supress_num / overall = {self.cond.sum().item()} / {unmasked_num * 512}")

        if normalize:
            delta_w /= delta_w.clone().norm(dim=-1, keepdim=True)
        
        return delta_w.unsqueeze(0)

    def get_image_features(self, images, norm=False):
        images = self.psp_preprocess(images)
        encodings, invert_img = self.model(images)
        # encodings = encodings[:, -1:]
        encodings = encodings.reshape(images.size(0), -1)

        # TODO: different from clip encodings, normalize may be harmful
        if norm:
            encodings /= encodings.clone().norm(dim=-1, keepdim=True)
        return encodings, invert_img
    
    def get_conditional_mask(self):
        if self.args.psp_loss_type == "multi_stage":
            return self.cond, None
        elif self.args.psp_loss_type == "dynamic":
            if self.args.delta_w_type == 'mean':
                delta_w = self.target_mean - self.source_mean
        else:
            raise RuntimeError(f"No psp loss whose type is {self.psp_loss_type} !")
        delta_w = delta_w.flatten()
        num_channel = len(delta_w)
        order = delta_w.abs().argsort()
        chosen_order = order[0:int(self.args.psp_alpha * num_channel)]
        # chosen_order = order[-int(self.args.psp_alpha * num_channel)::]  # Choose most important channels
        cond = torch.zeros(num_channel).to(self.device)
        cond[chosen_order] = 1
        cond = cond.unsqueeze(0)
        delta_w = delta_w.unsqueeze(0)
        return cond, delta_w

    def update_queue(self, src_vec, tgt_vec):
        if len(self.target_set) < self.args.sliding_window_size:
            self.source_set.append(src_vec.mean(0).detach())
            self.target_set.append(tgt_vec.mean(0).detach())
        else:
            self.source_set[self.source_pos] = src_vec.mean(0).detach()
            self.source_pos = (self.source_pos + 1) % self.args.sliding_window_size
            self.target_set[self.target_pos] = tgt_vec.mean(0).detach()
            self.target_pos = (self.target_pos + 1) % self.args.sliding_window_size

    def multi_stage_loss(self, target_encodings, source_encodings):
        if self.cond is not None:
            target_encodings = self.cond * target_encodings
            source_encodings = self.cond * source_encodings
        return F.l1_loss(target_encodings, source_encodings)
        
    def constrained_loss(self, cond):
        return torch.abs(cond.mean(1)-self.args.psp_alpha).mean()

    def cosine_similarity(self, vec1, vec2):
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()
        vec1 = vec1 / vec1.norm()
        vec2 = vec2 / vec2.norm()
        return (vec1 * vec2).sum()

    def update_w(self, source_encodings, target_encodings):
        if self.args.delta_w_type == 'mean':
            self.update_queue(source_encodings, target_encodings)
            self.source_mean = torch.stack(self.source_set).mean(0, keepdim=True)
            self.target_mean = torch.stack(self.target_set).mean(0, keepdim=True)
            # Get the editing direction
            delta_w = self.target_mean - self.source_mean
        return delta_w

    def dynamic_loss(self, target_encodings, source_encodings, delta_w):
        # Get the conditional vector to mask special enough channels
        delta_w = delta_w.flatten()
        num_channel = len(delta_w)
        order = delta_w.abs().argsort()
        chosen_order = order[0:int(self.args.psp_alpha * num_channel)]
        # chosen_order = order[-int(self.args.psp_alpha * num_channel)::]  # Choose most important channels
        cond = torch.zeros(num_channel).to(self.device)
        cond[chosen_order] = 1
        cond = cond.unsqueeze(0)

        # Get masked encodings
        target_encodings = cond * target_encodings
        source_encodings = cond * source_encodings

        # Update the mean direction of target domain and difference
        self.iter_diff.append(torch.abs(cond - self.cond).sum().cpu().item() / len(delta_w))
        self.iter_mean.append(cond.mean().cpu().item())
        self.iter_sim.append(self.cosine_similarity(delta_w, self.target_direction).sum().cpu().item())

        
        loss =  F.l1_loss(target_encodings, source_encodings)
        return loss

    def forward(self, target_imgs, source_imgs, iters=0, return_codes=False):
        if self.args.dataset == 'car':
            target_imgs = target_imgs[:, :, 64:448, :].contiguous()
            source_imgs = source_imgs[:, :, 64:448, :].contiguous()
        target_encodings, _ = self.get_image_features(target_imgs)
        source_encodings, _ = self.get_image_features(source_imgs)

        # Mask w+ codes controlling style and fine details
        if self.args.num_keep_first > 0:
            keep_num = self.args.num_keep_first * 512
            target_encodings = target_encodings[:, 0:keep_num]
            source_encodings = source_encodings[:, 0:keep_num]
        
        if self.args.psp_loss_type == "multi_stage":
            # edit_direction = target_encodings - source_encodings
            # theta = (edit_direction.clone() * self.target_direction).sum(dim=-1, keepdim=True)
            # return F.l1_loss(edit_direction, theta * self.target_direction)
            loss = self.multi_stage_loss(target_encodings, source_encodings)
        elif self.args.psp_loss_type == "dynamic":
            delta_w = self.update_w(source_encodings, target_encodings)
            regular_weight = max(0, \
                    (iters - self.args.sliding_window_size) / (self.args.iter - self.args.sliding_window_size))
            loss = regular_weight * self.dynamic_loss(target_encodings, source_encodings, delta_w=delta_w)
        else:
            raise RuntimeError(f"No psp loss whose type is {self.psp_loss_type} !")
        
        if return_codes:
            return loss, [target_encodings.detach(), source_encodings.detach()]
        else:
            return loss
        
        