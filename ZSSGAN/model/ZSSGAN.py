import sys
import os
import tqdm
from utils.file_utils import save_images

from utils.training_utils import mixing_noise

sys.path.insert(0, os.path.abspath('../'))


import torch


from model.sg2_model import Generator, Discriminator
from criteria.clip_loss import CLIPLoss 
from criteria.psp_loss import PSPLoss
from criteria.regularize_loss import RegularizeLoss      

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

class SG2Generator(torch.nn.Module):
    def __init__(self, checkpoint_path, latent_size=512, map_layers=8, img_size=256, channel_multiplier=2, device='cuda'):
        super(SG2Generator, self).__init__()

        self.generator = Generator(
            img_size, latent_size, map_layers, channel_multiplier=channel_multiplier
        ).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.generator.load_state_dict(checkpoint["g_ema"], strict=False)

        with torch.no_grad():
            self.mean_latent = self.generator.mean_latent(4096)

    def get_all_layers(self):
        return list(self.generator.children())

    def get_training_layers(self, phase):

        if phase == 'texture':
            # learned constant + first convolution + layers 3-10
            return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][2:10])   
        if phase == 'shape':
            # layers 1-2
             return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][0:2])
        if phase == 'no_fine':
            # const + layers 1-10
             return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][:10])
        if phase == 'shape_expanded':
            # const + layers 1-10
             return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][0:3])
        if phase == 'all':
            # everything, including mapping and ToRGB
            return self.get_all_layers() 
        else: 
            # everything except mapping and ToRGB
            return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][:])
            # Same with that when auto_layer_iters > 0 and auto_layer_k = 18
            # return list(self.get_all_layers())[2:4] + list(self.get_all_layers()[4][:])  

    def freeze_layers(self, layer_list=None):
        '''
        Disable training for all layers in list.
        '''
        if layer_list is None:
            self.freeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, False)

    def unfreeze_layers(self, layer_list=None):
        '''
        Enable training for all layers in list.
        '''
        if layer_list is None:
            self.unfreeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, True)

    def style(self, styles):
        '''
        Convert z codes to w codes.
        '''
        styles = [self.generator.style(s) for s in styles]
        return styles

    def get_s_code(self, styles, input_is_latent=False):
        return self.generator.get_s_code(styles, input_is_latent)

    def modulation_layers(self):
        return self.generator.modulation_layers

    # TODO Maybe convert to kwargs
    def forward(self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        input_is_s_code=False,
        noise=None,
        randomize_noise=True,
        delta_w=None):
        return self.generator(styles, return_latents=return_latents, truncation=truncation, \
            truncation_latent=self.mean_latent, noise=noise, randomize_noise=randomize_noise, \
                input_is_latent=input_is_latent, input_is_s_code=input_is_s_code, \
                    delta_w=delta_w)

class SG2Discriminator(torch.nn.Module):
    def __init__(self, checkpoint_path, img_size=256, channel_multiplier=2, device='cuda:0'):
        super(SG2Discriminator, self).__init__()

        self.discriminator = Discriminator(
            img_size, channel_multiplier=channel_multiplier
        ).to(device)

        checkpoint = torch.load(checkpoint_path, map_location=device)

        self.discriminator.load_state_dict(checkpoint["d"], strict=True)

    def get_all_layers(self):
        return list(self.discriminator.children())

    def get_training_layers(self):
        return self.get_all_layers() 

    def freeze_layers(self, layer_list=None):
        '''
        Disable training for all layers in list.
        '''
        if layer_list is None:
            self.freeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, False)

    def unfreeze_layers(self, layer_list=None):
        '''
        Enable training for all layers in list.
        '''
        if layer_list is None:
            self.unfreeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, True)

    def forward(self, images):
        return self.discriminator(images)

class ZSSGAN(torch.nn.Module):
    def __init__(self, args):
        super(ZSSGAN, self).__init__()

        self.args = args

        self.device = 'cuda'

        # Set up frozen (source) generator
        self.generator_frozen = SG2Generator(args.frozen_gen_ckpt, img_size=args.size).to(self.device)
        self.generator_frozen.freeze_layers()
        self.generator_frozen.eval()

        # Set up trainable (target) generator
        self.generator_trainable = SG2Generator(args.train_gen_ckpt, img_size=args.size).to(self.device)
        self.generator_trainable.freeze_layers()
        self.generator_trainable.unfreeze_layers(self.generator_trainable.get_training_layers(args.phase))
        self.generator_trainable.train()

        # Losses
        self.has_clip_loss = (sum(args.clip_model_weights) > 0)
        if self.has_clip_loss:
            self.clip_loss_models = {model_name: CLIPLoss(self.device, 
                                                        lambda_direction=args.lambda_direction, 
                                                        lambda_patch=args.lambda_patch, 
                                                        lambda_global=args.lambda_global, 
                                                        lambda_manifold=args.lambda_manifold, 
                                                        lambda_texture=args.lambda_texture,
                                                        clip_model=model_name,
                                                        args=args) 
                                    for model_name in args.clip_models}

            self.clip_model_weights = {model_name: weight for model_name, weight in zip(args.clip_models, args.clip_model_weights)}
            if self.args.source_type == 'online-prompt':
                self.args.online_clip_mean = {model_name: torch.zeros(512, dtype=torch.float16, device=self.device)\
                     for model_name in self.args.clip_models}
                self.get_samples_clip_mean(num=1000)
        self.has_psp_loss = args.psp_model_weight > 0
        if self.has_psp_loss:
            self.psp_loss_model = PSPLoss(self.device, args=args)
        
        # Style loss based on VGG16
        # self.vgg_loss_model = VGGLoss(self.device, args=args)
        self.mse_loss  = torch.nn.MSELoss()

        self.regularize_loss = RegularizeLoss(self.device,
                                            lambda_across=args.lambda_across,
                                            lambda_within=args.lambda_within,
                                            clip_model=args.clip_models[0],
                                            args=args)

        self.source_class = args.source_class
        self.target_class = args.target_class

        self.auto_layer_k     = args.auto_layer_k
        self.auto_layer_iters = args.auto_layer_iters

        if args.target_img_list is not None and self.has_clip_loss:
            self.set_img2img_direction()
            # self.vgg_loss_model.compute_target_features(args.target_img_list)

    
    def get_samples_clip_mean(self, num=1000, debug=False):
        '''
        Get mean value of frozen gan's samples
        '''
        with torch.no_grad():
            for i in tqdm.tqdm(range(num // self.args.batch)):
                sample_z = mixing_noise(self.args.batch, 512, self.args.mixing, self.device)
                sampled_src = self.generator_frozen(sample_z, truncation=self.args.sample_truncation)[0]
                for k in self.args.online_clip_mean.keys():
                    self.args.online_clip_mean[k] += self.clip_loss_models[k].get_image_features(sampled_src).sum(dim=0)
                if debug:
                    save_images(sampled_src, self.args.output_dir, 'sample', 1, i)
            for k in self.args.online_clip_mean.keys():
                self.args.online_clip_mean[k] /= num

    def set_img2img_direction(self):
        with torch.no_grad():
            sample_z  = torch.randn(self.args.img2img_batch, 512, device=self.device)
            generated = self.generator_trainable([sample_z])[0]

            for _, model in self.clip_loss_models.items():
                direction = model.compute_img2img_direction(generated, self.args.target_img_list)

                model.target_direction = direction

    def determine_opt_layers(self):

        sample_z = torch.randn(self.args.auto_layer_batch, 512, device=self.device)

        initial_w_codes = self.generator_frozen.style([sample_z])
        initial_w_codes = initial_w_codes[0].unsqueeze(1).repeat(1, self.generator_frozen.generator.n_latent, 1)

        w_codes = torch.Tensor(initial_w_codes.cpu().detach().numpy()).to(self.device)

        w_codes.requires_grad = True

        w_optim = torch.optim.SGD([w_codes], lr=0.01)

        for _ in range(self.auto_layer_iters):
            w_codes_for_gen = w_codes.unsqueeze(0)
            generated_from_w = self.generator_trainable(w_codes_for_gen, input_is_latent=True)[0]

            w_loss = [self.clip_model_weights[model_name] * self.clip_loss_models[model_name].global_clip_loss(generated_from_w, self.target_class) for model_name in self.clip_model_weights.keys()]
            w_loss = torch.sum(torch.stack(w_loss))
            
            w_optim.zero_grad()
            w_loss.backward()
            w_optim.step()
        
        layer_weights = torch.abs(w_codes - initial_w_codes).mean(dim=-1).mean(dim=0)
        chosen_layer_idx = torch.topk(layer_weights, self.auto_layer_k)[1].cpu().numpy()

        all_layers = list(self.generator_trainable.get_all_layers())

        conv_layers = list(all_layers[4])
        rgb_layers = list(all_layers[6]) # currently not optimized

        idx_to_layer = all_layers[2:4] + conv_layers # add initial convs to optimization

        chosen_layers = [idx_to_layer[idx] for idx in chosen_layer_idx] 

        # uncomment to add RGB layers to optimization.

        # for idx in chosen_layer_idx:
        #     if idx % 2 == 1 and idx >= 3 and idx < 14:
        #         chosen_layers.append(rgb_layers[(idx - 3) // 2])

        # uncomment to add learned constant to optimization
        # chosen_layers.append(all_layers[1])
                
        return chosen_layers

    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
        delta_w=None,
        iters=0,
        return_psp_codes=False,
    ):
    
        if self.training and self.auto_layer_iters > 0 and self.has_clip_loss:
            self.generator_trainable.unfreeze_layers()
            train_layers = self.determine_opt_layers()

            if not isinstance(train_layers, list):
                train_layers = [train_layers]

            self.generator_trainable.freeze_layers()
            self.generator_trainable.unfreeze_layers(train_layers)
        
        with torch.no_grad():
            if input_is_latent:
                w_styles = styles
            else:
                w_styles = self.generator_frozen.style(styles)
            
            if self.args.return_w_only:
                return w_styles
            
            frozen_img = self.generator_frozen(w_styles, input_is_latent=True, \
                truncation=truncation, randomize_noise=randomize_noise, \
                    delta_w=delta_w)[0]

        trainable_img = self.generator_trainable(w_styles, input_is_latent=True, \
            truncation=truncation, randomize_noise=randomize_noise)[0]
        
        loss = 0.0
        if self.has_clip_loss:
            loss += torch.sum(torch.stack([self.clip_model_weights[model_name] * self.clip_loss_models[model_name](frozen_img, \
                self.source_class, trainable_img, self.target_class) for model_name in self.clip_model_weights.keys()]))
        psp_codes = None
        if self.has_psp_loss:
            psp_loss, psp_codes = self.psp_loss_model(trainable_img, frozen_img, iters=iters, return_codes=True)
            loss += self.args.psp_model_weight * psp_loss

        # loss += 2 * self.vgg_loss_model(trainable_img)
        if return_psp_codes:
            return [frozen_img, trainable_img], loss, psp_codes
        else:
            return [frozen_img, trainable_img], loss

    def pivot(self):
        par_frozen = dict(self.generator_frozen.named_parameters())
        par_train  = dict(self.generator_trainable.named_parameters())

        for k in par_frozen.keys():
            par_frozen[k] = par_train[k]
